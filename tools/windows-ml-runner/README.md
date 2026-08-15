# Windows ML Runner

This is the Windows-only execution boundary used by the Rust adapter. It is
published separately for `win-x64` and `win-arm64` and must be built with the
checked-in .NET SDK and NuGet lock file.

```powershell
dotnet restore WindowsMlRunner.csproj --locked-mode
dotnet publish WindowsMlRunner.csproj --configuration Release --runtime win-x64 --no-restore --self-contained false --output out/win-x64
dotnet publish WindowsMlRunner.csproj --configuration Release --runtime win-arm64 --no-restore --self-contained false --output out/win-arm64
```

The runner calls `ExecutionProviderCatalog.RegisterCertifiedAsync`, obtains the
actual `OrtEpDevice`, appends that device to `SessionOptions`, creates an
`InferenceSession`, and validates runtime metadata before publishing raw output.
The report is diagnostic evidence only; no target runner means no support or
performance claim.

The locked package graph intentionally uses `Microsoft.WindowsAppSDK.ML` and
`Microsoft.Windows.AI.MachineLearning` at `2.1.74` together with
`WindowsAppSDKSelfContained=true`. The ML package's official `2.1.74` metadata
requires the matching Windows ML projection (`[2.1.74, 3.0.0)`),
`Microsoft.WindowsAppSDK.Base` `2.0.4`, and Foundation `2.0.21`. The separate
`Microsoft.WindowsAppSDK.Runtime` `2.1.3` meta-package is excluded because its
`ComponentReference` target requires the ML component at `2.1.1`; including it
creates an internally inconsistent graph and fails before compilation. The
self-contained flag packages the locked Windows App SDK components with this
task-local runner and does not claim an installed product runtime or Windows
ARM64 support.

For task-local performance capture, add `"performanceRuns": 30` to the runner
request. The first inference remains the fixed cold run; the runner then uses
the frozen five warmup runs before recording 30 samples on the same
`SessionOptions`/`InferenceSession`. The report includes `warmupRuns`,
`initializationMs` (ending before cold inference), `coldInferenceMs`, and
`peakProcessRssBytes`. The default is zero, so normal adapter requests retain
one smoke/inference execution without performance warmups or samples.
