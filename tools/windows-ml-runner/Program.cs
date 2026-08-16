using System.Buffers.Binary;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;

internal static class Program
{
    private const string SourcePackage = "Microsoft.WindowsAppSDK.ML";
    private const string RuntimePackage = "Microsoft.Windows.AI.MachineLearning";
    private const string PackageVersion = "2.1.74";
    private const int PerformanceWarmupRuns = 5;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    public static async Task<int> Main(string[] args)
    {
        string? reportPath = FindOption(args, "--report");
        var state = new RunState();
        try
        {
            string requestPath = RequiredOption(args, "--request");
            reportPath = RequiredOption(args, "--report");
            RunnerRequest request = JsonSerializer.Deserialize<RunnerRequest>(await File.ReadAllTextAsync(requestPath), JsonOptions)
                ?? throw new ArgumentException("runner request is empty");
            RunnerReport report = await RunAsync(request, state);
            await WriteReportAsync(reportPath, report);
            return 0;
        }
        catch (Exception error)
        {
            if (!string.IsNullOrWhiteSpace(reportPath))
            {
                try
                {
                    await WriteReportAsync(reportPath, FailureReport(state, error));
                }
                catch (Exception reportError)
                {
                    Console.Error.WriteLine($"Windows ML failure report could not be written: {reportError}");
                }
            }
            Console.Error.WriteLine(error);
            return 1;
        }
        finally
        {
            CleanupProfile(state.ProfilePath, state.ProfilePrefix);
        }
    }

    private static async Task<RunnerReport> RunAsync(RunnerRequest request, RunState state)
    {
        state.FailureStage = "platform-validation";
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("Windows ML must run on Windows; Linux ORT is not an accepted substitute.");
        }
        string target = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => "win-x64",
            Architecture.Arm64 => "win-arm64",
            _ => throw new PlatformNotSupportedException("Only Windows x64 and ARM64 are valid Windows ML targets."),
        };

        ValidateRequest(request);
        state.FailureStage = "model-identity";
        string modelSha256 = Sha256File(request.ModelPath);
        if (!modelSha256.Equals(request.ExpectedModelSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"model SHA-256 mismatch: expected {request.ExpectedModelSha256}, got {modelSha256}");
        }

        state.FailureStage = "input-validation";
        var inputBindings = request.Inputs.Select(binding =>
        {
            string name = binding.Name ?? throw new InvalidDataException($"input role {binding.Role} has no model feature name");
            float[] values = ReadFloat32LittleEndian(binding.Path, ElementCount(binding.Shape));
            if (values.Any(value => !float.IsFinite(value))) throw new InvalidDataException($"input role {binding.Role} contains a non-finite value");
            return (binding, name, values);
        }).ToArray();

        Stopwatch initializationTimer = Stopwatch.StartNew();
        state.FailureStage = "catalog-registration";
        state.WindowsMlApiCalled = true;
        state.CatalogRegistrationAttempted = true;
        ExecutionProviderCatalog catalog = ExecutionProviderCatalog.GetDefault();
        await catalog.RegisterCertifiedAsync();
        state.CatalogRegistrationCompleted = true;
        double catalogRegistrationEndMs = initializationTimer.Elapsed.TotalMilliseconds;

        state.FailureStage = "device-selection";
        OrtEnv ortEnv = OrtEnv.Instance();
        IReadOnlyList<OrtEpDevice> availableDevices = ortEnv.GetEpDevices();
        if (availableDevices.Count == 0) throw new InvalidOperationException("Windows ML reported no execution-provider devices.");
        OrtEpDevice selectedDevice = availableDevices
            .OrderBy(device => DevicePriority(device.HardwareDevice.Type))
            .ThenBy(device => device.EpName, StringComparer.Ordinal)
            .First();
        double deviceSelectionEndMs = initializationTimer.Elapsed.TotalMilliseconds;

        using var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        if (request.CollectExecutionProfile)
        {
            state.ProfilePrefix = Path.Combine(Path.GetTempPath(), $"rimeflow-winml-{Environment.ProcessId}-{Guid.NewGuid():N}-");
            sessionOptions.ProfileOutputPathPrefix = state.ProfilePrefix;
            sessionOptions.EnableProfiling = true;
        }
        sessionOptions.AppendExecutionProvider(ortEnv, [selectedDevice], new Dictionary<string, string>());
        double sessionOptionsEndMs = initializationTimer.Elapsed.TotalMilliseconds;

        state.FailureStage = "session-creation";
        using var session = new InferenceSession(request.ModelPath, sessionOptions);
        state.SessionCreated = true;
        double sessionCreationEndMs = initializationTimer.Elapsed.TotalMilliseconds;

        state.FailureStage = "metadata-validation";
        var inputMetadata = request.Inputs.Select(binding => ResolveMetadata(session.InputMetadata, binding, "input")).ToArray();
        var outputMetadata = request.Outputs.Select(binding => ResolveMetadata(session.OutputMetadata, binding, "output")).ToArray();
        if (inputMetadata.Length != inputBindings.Length) throw new InvalidDataException("input binding count changed during metadata validation");
        double metadataValidationEndMs = initializationTimer.Elapsed.TotalMilliseconds;
        initializationTimer.Stop();

        Stopwatch inputBindingTimer = Stopwatch.StartNew();
        var namedInputs = inputBindings.Select((item, index) =>
            NamedOnnxValue.CreateFromTensor(item.name, new DenseTensor<float>(item.values, inputMetadata[index].Metadata.Dimensions.ToArray()))).ToArray();
        inputBindingTimer.Stop();

        state.FailureStage = "inference";
        Stopwatch coldInferenceTimer = Stopwatch.StartNew();
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(namedInputs);
        coldInferenceTimer.Stop();
        state.InferenceExecuted = true;

        state.FailureStage = "output-validation";
        var outputSnapshots = new List<OutputSnapshot>();
        for (int index = 0; index < request.Outputs.Length; index++)
        {
            TensorBinding binding = request.Outputs[index];
            (string name, NodeMetadata metadata) expected = outputMetadata[index];
            DisposableNamedOnnxValue result = results.SingleOrDefault(item => item.Name.Equals(expected.name, StringComparison.Ordinal))
                ?? throw new InvalidDataException($"runtime output {expected.name} was not produced");
            float[] values = result.AsEnumerable<float>().ToArray();
            if (values.Length != ElementCount(binding.Shape) || values.Any(value => !float.IsFinite(value)))
            {
                throw new InvalidDataException($"output role {binding.Role} has an element or finite-value mismatch");
            }
            WriteFloat32LittleEndian(binding.Path, values);
            outputSnapshots.Add(new OutputSnapshot(binding.Role, expected.name, binding.Shape, values.Length, binding.Path));
        }

        for (int index = 0; index < (request.PerformanceRuns > 0 ? PerformanceWarmupRuns : 0); index++)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> warmupResults = session.Run(namedInputs);
            ValidateOutputResults(warmupResults, request.Outputs, outputMetadata);
        }

        var warmInferenceSamples = new List<double>(request.PerformanceRuns);
        for (int index = 0; index < request.PerformanceRuns; index++)
        {
            Stopwatch warmInferenceTimer = Stopwatch.StartNew();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> warmResults = session.Run(namedInputs);
            warmInferenceTimer.Stop();
            ValidateOutputResults(warmResults, request.Outputs, outputMetadata);
            warmInferenceSamples.Add(warmInferenceTimer.Elapsed.TotalMilliseconds);
        }

        state.FailureStage = "provider-introspection";
        IReadOnlyList<OrtEpDevice?> inputDevices = session.GetEpDeviceForInputs();
        if (inputDevices.Count != inputMetadata.Length || inputDevices.Any(device => device is null))
        {
            throw new InvalidDataException("Windows ML did not expose the actual EP device for every input");
        }
        string[] profileProviders = [];
        if (request.CollectExecutionProfile)
        {
            string profilePath = session.EndProfiling();
            state.ProfilePath = profilePath;
            profileProviders = ReadProfileProviders(profilePath);
            if (profileProviders.Length == 0) throw new InvalidDataException("Windows ML profile did not expose a provider");
        }

        state.FailureStage = "module-identity";
        Assembly catalogAssembly = typeof(ExecutionProviderCatalog).Assembly;
        Assembly ortAssembly = typeof(InferenceSession).Assembly;
        // The NuGet package ships the native WinRT component separately from
        // the managed projection that contains ExecutionProviderCatalog.
        if (!catalogAssembly.GetName().Name!.Equals("Microsoft.Windows.AI.MachineLearning.Projection", StringComparison.Ordinal) ||
            !ortAssembly.GetName().Name!.Equals("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("loaded assemblies are not the Windows ML projection and its bundled ORT API");
        }

        state.FailureStage = "dependency-identity";
        state.FailureStage = "sdk-runtime-identity";
        state.RuntimeIntrospectionComplete = true;
        state.FailureStage = "artifact-publication";
        state.OutputPublished = true;

        return new RunnerReport
        {
            SchemaVersion = 1,
            State = "runtime-verified",
            Target = target,
            RuntimeExecuted = true,
            FailureStage = state.FailureStage,
            WindowsMlApiCalled = state.WindowsMlApiCalled,
            CatalogRegistrationAttempted = state.CatalogRegistrationAttempted,
            CatalogRegistrationCompleted = state.CatalogRegistrationCompleted,
            SessionCreated = state.SessionCreated,
            InferenceExecuted = state.InferenceExecuted,
            RuntimeIntrospectionComplete = state.RuntimeIntrospectionComplete,
            OutputPublished = state.OutputPublished,
            Runtime = new RuntimeIdentity
            {
                SourcePackage = new PackageIdentity { Id = SourcePackage, Version = PackageVersion },
                RuntimePackage = new PackageIdentity { Id = RuntimePackage, Version = PackageVersion },
                OrtVersion = ortEnv.GetVersionString(),
            },
            Execution = new ExecutionIdentity
            {
                SelectedDevice = DeviceSnapshot(selectedDevice),
                ExecutionProfileCollected = request.CollectExecutionProfile,
                ProfileProviders = profileProviders,
                SessionInputDevices = inputDevices.Select(device => DeviceSnapshot(device!)).ToArray(),
            },
            Outputs = outputSnapshots.ToArray(),
            Performance = request.PerformanceRuns > 0
                ? new PerformanceIdentity
                {
                    WarmupRuns = PerformanceWarmupRuns,
                    InitializationMs = initializationTimer.Elapsed.TotalMilliseconds,
                    InitializationBreakdownMs = new InitializationBreakdown
                    {
                        CatalogRegistration = catalogRegistrationEndMs,
                        DeviceSelection = deviceSelectionEndMs - catalogRegistrationEndMs,
                        SessionOptions = sessionOptionsEndMs - deviceSelectionEndMs,
                        SessionCreation = sessionCreationEndMs - sessionOptionsEndMs,
                        MetadataValidation = metadataValidationEndMs - sessionCreationEndMs,
                        InputBindingExcluded = inputBindingTimer.Elapsed.TotalMilliseconds,
                    },
                    ColdInferenceMs = coldInferenceTimer.Elapsed.TotalMilliseconds,
                    WarmInferenceMs = warmInferenceSamples.ToArray(),
                    PeakProcessRssBytes = Process.GetCurrentProcess().PeakWorkingSet64,
                }
                : null,
        };
    }

    private static void ValidateRequest(RunnerRequest request)
    {
        if (request.SchemaVersion != 1) throw new ArgumentException("unsupported Windows ML runner request schema");
        if (request.Mode is not ("smoke" or "infer")) throw new ArgumentException("runner mode must be smoke or infer");
        if (request.PerformanceRuns is < 0 or > 100) throw new ArgumentException("performanceRuns must be between 0 and 100");
        if (request.Inputs is null || request.Inputs.Length != 1) throw new ArgumentException("Windows ML base adapter requires one input");
        if (request.Outputs is null || request.Outputs.Length == 0) throw new ArgumentException("Windows ML base adapter requires outputs");
        if (string.IsNullOrWhiteSpace(request.ExpectedModelSha256) || request.ExpectedModelSha256.Length != 64 || request.ExpectedModelSha256.Any(character => !Uri.IsHexDigit(character))) throw new ArgumentException("invalid expected model SHA-256");
        foreach (TensorBinding binding in request.Inputs.Concat(request.Outputs))
        {
            if (string.IsNullOrWhiteSpace(binding.Role) || binding.Shape is null || binding.Shape.Length == 0 || binding.Shape.Any(dimension => dimension <= 0) || binding.Dtype != "f32" || string.IsNullOrWhiteSpace(binding.Path) || (binding.Name is null && binding.Index is null))
            {
                throw new ArgumentException($"invalid Windows ML tensor binding for role {binding.Role}");
            }
        }
    }

    private static (string name, NodeMetadata Metadata) ResolveMetadata(IReadOnlyDictionary<string, NodeMetadata> metadata, TensorBinding binding, string direction)
    {
        string name;
        if (binding.Name is not null)
        {
            name = binding.Name;
            if (!metadata.ContainsKey(name)) throw new InvalidDataException($"manifest {direction} role {binding.Role} name {name} is absent from runtime metadata");
        }
        else
        {
            int index = binding.Index!.Value;
            if (index < 0 || index >= metadata.Count) throw new InvalidDataException($"manifest {direction} role {binding.Role} index {index} is absent from runtime metadata");
            name = metadata.Keys.ElementAt(index);
        }
        NodeMetadata value = metadata[name];
        if (!value.IsTensor || value.ElementDataType != TensorElementType.Float || !value.Dimensions.SequenceEqual(binding.Shape))
        {
            throw new InvalidDataException($"runtime {direction} metadata drift for role {binding.Role}");
        }
        return (name, value);
    }

    private static void ValidateOutputResults(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        IReadOnlyList<TensorBinding> bindings,
        IReadOnlyList<(string name, NodeMetadata Metadata)> metadata)
    {
        for (int index = 0; index < bindings.Count; index++)
        {
            TensorBinding binding = bindings[index];
            string expectedName = metadata[index].name;
            DisposableNamedOnnxValue result = results.SingleOrDefault(item => item.Name.Equals(expectedName, StringComparison.Ordinal))
                ?? throw new InvalidDataException($"runtime output {expectedName} was not produced");
            float[] values = result.AsEnumerable<float>().ToArray();
            if (values.Length != ElementCount(binding.Shape) || values.Any(value => !float.IsFinite(value)))
            {
                throw new InvalidDataException($"output role {binding.Role} has an element or finite-value mismatch");
            }
        }
    }

    private static int ElementCount(IReadOnlyList<int> shape) => shape.Aggregate(1, checked((count, dimension) => count * dimension));

    private static int DevicePriority(OrtHardwareDeviceType type) => type switch
    {
        OrtHardwareDeviceType.NPU => 0,
        OrtHardwareDeviceType.GPU => 1,
        OrtHardwareDeviceType.CPU => 2,
        _ => 3,
    };

    private static DeviceIdentity DeviceSnapshot(OrtEpDevice device) => new()
    {
        EpName = device.EpName,
        EpVendor = device.EpVendor,
        Hardware = new HardwareIdentity { Type = device.HardwareDevice.Type.ToString() },
    };

    private static string[] ReadProfileProviders(string profilePath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(profilePath));
        return document.RootElement.EnumerateArray()
            .Where(item => item.TryGetProperty("args", out JsonElement args) && args.TryGetProperty("provider", out _))
            .Select(item => item.GetProperty("args").GetProperty("provider").GetString())
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Select(value => value!)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();
    }

    private static float[] ReadFloat32LittleEndian(string path, int expectedElements)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != expectedElements * sizeof(float)) throw new InvalidDataException($"expected {expectedElements * sizeof(float)} input bytes, got {bytes.Length}");
        var values = new float[expectedElements];
        for (int index = 0; index < values.Length; index++) values[index] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float))));
        return values;
    }

    private static void WriteFloat32LittleEndian(string path, IReadOnlyList<float> values)
    {
        byte[] bytes = new byte[values.Count * sizeof(float)];
        for (int index = 0; index < values.Count; index++) BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float)), BitConverter.SingleToInt32Bits(values[index]));
        File.WriteAllBytes(path, bytes);
    }

    private static RunnerReport FailureReport(RunState state, Exception error) => new()
    {
        SchemaVersion = 1,
        State = "failed",
        RuntimeExecuted = state.WindowsMlApiCalled,
        FailureStage = state.FailureStage,
        WindowsMlApiCalled = state.WindowsMlApiCalled,
        CatalogRegistrationAttempted = state.CatalogRegistrationAttempted,
        CatalogRegistrationCompleted = state.CatalogRegistrationCompleted,
        SessionCreated = state.SessionCreated,
        InferenceExecuted = state.InferenceExecuted,
        RuntimeIntrospectionComplete = state.RuntimeIntrospectionComplete,
        OutputPublished = false,
        Error = new RunnerError { Type = error.GetType().FullName, Message = error.Message },
    };

    private static async Task WriteReportAsync(string path, RunnerReport report)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(report, JsonOptions) + Environment.NewLine);
    }

    private static void CleanupProfile(string? profilePath, string? profilePrefix)
    {
        if (!string.IsNullOrWhiteSpace(profilePath)) TryDelete(profilePath);
        if (string.IsNullOrWhiteSpace(profilePrefix)) return;
        string directory = Path.GetDirectoryName(profilePrefix)!;
        if (!Directory.Exists(directory)) return;
        foreach (string path in Directory.EnumerateFiles(directory, $"{Path.GetFileName(profilePrefix)}*")) TryDelete(path);
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { }
    }

    private static string Sha256File(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static string RequiredOption(string[] args, string name) => FindOption(args, name) ?? throw new ArgumentException($"missing {name}");

    private static string? FindOption(string[] args, string name)
    {
        int index = Array.IndexOf(args, name);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    private sealed class RunState
    {
        public string FailureStage { get; set; } = "argument-validation";
        public bool WindowsMlApiCalled { get; set; }
        public bool CatalogRegistrationAttempted { get; set; }
        public bool CatalogRegistrationCompleted { get; set; }
        public bool SessionCreated { get; set; }
        public bool InferenceExecuted { get; set; }
        public bool RuntimeIntrospectionComplete { get; set; }
        public bool OutputPublished { get; set; }
        public string? ProfilePrefix { get; set; }
        public string? ProfilePath { get; set; }
    }

    private sealed class RunnerRequest
    {
        public int SchemaVersion { get; set; }
        public string Mode { get; set; } = "";
        public string ModelPath { get; set; } = "";
        public string ExpectedModelSha256 { get; set; } = "";
        public int PerformanceRuns { get; set; }
        public bool CollectExecutionProfile { get; set; } = true;
        public TensorBinding[] Inputs { get; set; } = [];
        public TensorBinding[] Outputs { get; set; } = [];
    }

    private sealed class TensorBinding
    {
        public string Role { get; set; } = "";
        public string? Name { get; set; }
        public int? Index { get; set; }
        public int[] Shape { get; set; } = [];
        public string Dtype { get; set; } = "";
        public string Path { get; set; } = "";
    }

    private sealed class RunnerReport
    {
        public int SchemaVersion { get; set; }
        public string State { get; set; } = "";
        public string? Target { get; set; }
        public bool RuntimeExecuted { get; set; }
        public string FailureStage { get; set; } = "";
        public bool WindowsMlApiCalled { get; set; }
        public bool CatalogRegistrationAttempted { get; set; }
        public bool CatalogRegistrationCompleted { get; set; }
        public bool SessionCreated { get; set; }
        public bool InferenceExecuted { get; set; }
        public bool RuntimeIntrospectionComplete { get; set; }
        public bool OutputPublished { get; set; }
        public RuntimeIdentity? Runtime { get; set; }
        public ExecutionIdentity? Execution { get; set; }
        public OutputSnapshot[]? Outputs { get; set; }
        public PerformanceIdentity? Performance { get; set; }
        public RunnerError? Error { get; set; }
    }

    private sealed class RuntimeIdentity
    {
        public PackageIdentity SourcePackage { get; set; } = new();
        public PackageIdentity RuntimePackage { get; set; } = new();
        public string OrtVersion { get; set; } = "";
    }

    private sealed class PackageIdentity
    {
        public string Id { get; set; } = "";
        public string Version { get; set; } = "";
    }

    private sealed class ExecutionIdentity
    {
        public DeviceIdentity? SelectedDevice { get; set; }
        public bool ExecutionProfileCollected { get; set; }
        public string[] ProfileProviders { get; set; } = [];
        public DeviceIdentity[] SessionInputDevices { get; set; } = [];
    }

    private sealed class PerformanceIdentity
    {
        public int WarmupRuns { get; set; }
        public double InitializationMs { get; set; }
        public InitializationBreakdown InitializationBreakdownMs { get; set; } = new();
        public double ColdInferenceMs { get; set; }
        public double[] WarmInferenceMs { get; set; } = [];
        public long PeakProcessRssBytes { get; set; }
    }

    private sealed class InitializationBreakdown
    {
        public double CatalogRegistration { get; set; }
        public double DeviceSelection { get; set; }
        public double SessionOptions { get; set; }
        public double SessionCreation { get; set; }
        public double MetadataValidation { get; set; }
        public double InputBindingExcluded { get; set; }
    }

    private sealed class DeviceIdentity
    {
        public string EpName { get; set; } = "";
        public string EpVendor { get; set; } = "";
        public HardwareIdentity Hardware { get; set; } = new();
    }

    private sealed class HardwareIdentity
    {
        public string Type { get; set; } = "";
    }

    private sealed record OutputSnapshot(string Role, string Name, IReadOnlyList<int> Shape, int Elements, string Path);

    private sealed class RunnerError
    {
        public string? Type { get; set; }
        public string Message { get; set; } = "";
    }
}
