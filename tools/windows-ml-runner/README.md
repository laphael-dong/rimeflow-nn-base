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
