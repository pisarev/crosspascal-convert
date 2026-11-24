{ ************************************************************************** }
{                                                                            }
{ CrossPascal.YoloBundle                                                      }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.YoloBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}
{$B-}

interface

uses
  CrossPascal.Json;

{ The request pins an original checkpoint, its weight-free graph recipe and
  the consumer runtime. Conversion never invokes Python or an exporter.
  The destination must be new. The returned receipt pins manifest.json. }
function ConvertYoloBundle(const RequestFile: string; const ValidateOnly: Boolean = False): TJSONObject;

implementation

uses
  {$IFDEF FPC}SysUtils, Classes{$IFNDEF UNIX}, Windows{$ENDIF}
  {$ELSE}System.SysUtils, System.Classes, Winapi.Windows{$ENDIF},
  TVM, TVM.CpuGraph, TVM.NativeFiles, TVM.Yolo.Profile, TVM.Yolo.Identity,
  CrossPascal.SHA256, CrossPascal.Onnx, CrossPascal.YoloSource,
  CrossPascal.TVMCompiler, CrossPascal.TVMOnnx, CrossPascal.NativeIO;

{$IFDEF UNIX}
{ Linux rename without replacing an existing destination, including an empty
  directory created by another process during compilation. }
function RenameNoReplace(OldDirectory: Integer; OldPath: PAnsiChar; NewDirectory: Integer;
  NewPath: PAnsiChar; Flags: Cardinal): Integer;
  cdecl; external 'c' name 'renameat2';
{$ENDIF}

function OpenRead(const FileName: string): TStream;
begin
  Result := TNativeReadStream.Create(FileName);
end;

function ReadJSONFile(const FileName: string): TJSONObject;
var
  Stream: TStream;
  Bytes: TBytes;
  Value: TJSONValue;
begin
  Stream := OpenRead(FileName);
  try
    NativeRequire((Stream.Size > 0) and (Stream.Size <= 2 * 1024 * 1024),
      'Invalid YOLO request or catalog byte size');
    SetLength(Bytes, Stream.Size);
    Stream.ReadBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
  Value := ParseCrossJSON(TEncoding.UTF8.GetString(Bytes));
  try
    NativeUniqueKeys(Value);
    Result := NativeObject(Value);
  except
    Value.Free;
    raise;
  end;
end;

function RequestPath(const Base: string; const Obj: TJSONObject; const Name: string): string;
var
  Value: string;
begin
  Value := NativeText(Obj, Name);
  NativeRequire((Value <> '') and (Pos(#0, Value) = 0), 'Invalid YOLO request path: ' + Name);
  if {$IFDEF UNIX}(Value[1] = '/'){$ELSE}(ExtractFileDrive(Value) <> '') or
    (Value[1] = '\') or (Value[1] = '/'){$ENDIF} then
      Result := ExpandFileName(Value)
  else
    Result := ExpandFileName(IncludeTrailingPathDelimiter(Base) + Value);
end;

procedure WriteJSONFile(const FileName: string; const Obj: TJSONObject);
var
  Stream: TStream;
  Bytes: TBytes;
begin
  Bytes := TEncoding.UTF8.GetBytes(Obj.ToJSON + #10);
  Stream := TNativeWriteStream.Create(FileName);
  try
    if Length(Bytes) > 0 then Stream.WriteBuffer(Bytes[0], Length(Bytes));
  finally
    Stream.Free;
  end;
end;

function FilePin(const FileName: string): TJSONObject;
var
  Stream: TStream;
begin
  Stream := OpenRead(FileName);
  try
    Result := TJSONObject.Create;
    try
      Result.AddPair('bytes', JSONInteger(Stream.Size));
      Result.AddPair('sha256', SHA256Stream(Stream));
    except
      Result.Free;
      raise;
    end;
  finally
    Stream.Free;
  end;
end;

function RuntimeFileName(const Role, TargetOS: string): string;
var
  Base: string;
begin
  if Role = 'ffi' then
    Base := 'tvm_ffi'
  else if Role = 'runtime' then
    Base := 'tvm_runtime'
  else
    Base := 'tvm_runtime_' + Role;
  if TargetOS = 'windows' then
    Result := Base + '.dll'
  else
    Result := 'lib' + Base + '.so';
end;

function CheckRuntime(const Directory, ManifestSHA256, TargetOS: string; const ManifestBytes: Int64;
  const NeedCuda: Boolean): TArray<INativeFileLease>;
var
  Root, Row: TJSONObject;
  Rows: TJSONArray;
  Role, Name: string;
  Roles: TArray<string>;
  I, J, FFI, Runtime, Extra, Cuda: Integer;
begin
  SetLength(Result, 1);
  Result[0] := HoldNativeFile(Directory, 'runtime.json', ManifestSHA256, ManifestBytes);
  Root := ReadNativeObject(Result[0]);
  try
    NativeRequire((NativeText(Root, 'schema') = 'crosspascal.native-runtime.v1') and
      (NativeText(Root, 'os') = TargetOS) and (NativeText(Root, 'arch') = 'x86_64') and
      (NativeText(Root, 'tvm') = '0.26.0') and (NativeText(Root, 'ffi_abi') = '0.1.14'),
      'The YOLO consumer runtime platform, schema or ABI differs');
    Rows := NativeArray(NativeField(Root, 'files'));
    NativeRequire((Rows.Count >= 3) and (Rows.Count <= 32), 'Invalid YOLO runtime file count');
    SetLength(Result, Rows.Count + 1);
    SetLength(Roles, Rows.Count);
    FFI := 0;
    Runtime := 0;
    Extra := 0;
    Cuda := 0;
    for I := 0 to Rows.Count - 1 do
    begin
      Row := NativeObject(Rows.Items[I]);
      Role := NativeText(Row, 'role');
      Name := NativeText(Row, 'file');
      if Role = 'ffi' then
      begin
        Inc(FFI);
        NativeRequire(Name = RuntimeFileName(Role, TargetOS), 'Unexpected YOLO runtime FFI filename');
      end
      else if Role = 'runtime' then
      begin
        Inc(Runtime);
        NativeRequire(Name = RuntimeFileName(Role, TargetOS), 'Unexpected YOLO runtime filename');
      end
      else if Role = 'extra' then
      begin
        Inc(Extra);
        NativeRequire(Name = RuntimeFileName(Role, TargetOS), 'Unexpected YOLO extra runtime filename');
      end
      else if Role = 'cuda' then
      begin
        Inc(Cuda);
        NativeRequire(Name = RuntimeFileName(Role, TargetOS), 'Unexpected YOLO CUDA runtime filename');
      end
      else
        NativeRequire(Role = 'dependency_cuda', 'Unknown YOLO runtime file role');
      Roles[I] := Role;
      Result[I + 1] := HoldNativeFile(Directory, Name, NativeText(Row, 'sha256'),
        NativeInteger(Row, 'bytes'));
      RequireNativeX64LibraryForOS(Result[I + 1], TargetOS);
      for J := 0 to I - 1 do
      begin
        NativeRequire(not SameFileName(Result[I + 1].FileName, Result[J + 1].FileName),
          'Duplicate YOLO runtime filename');
        NativeRequire((Role = 'dependency_cuda') or (Role <> Roles[J]), 'Duplicate YOLO runtime role');
      end;
    end;
    NativeRequire((FFI = 1) and (Runtime = 1) and (Extra = 1) and
      ((not NeedCuda) or (Cuda = 1)), 'The YOLO consumer runtime is incomplete');
  finally
    Root.Free;
  end;
end;

procedure CheckGraph(const Graph: TOnnxModel; const Profile: TYoloProfile);
var
  I, J, Count: Integer;
  Value: TOnnxValue;
  Initializer: TOnnxTensor;
  IsInitializer: Boolean;
begin
  Count := 0;
  for Value in Graph.Inputs do
  begin
    IsInitializer := False;
    for Initializer in Graph.Initializers do
      if Value.Name = Initializer.Name then IsInitializer := True;
    if IsInitializer then Continue;
    Inc(Count);
    NativeRequire((Value.Name = Profile.InputName) and (Value.DataType = OnnxFloat) and
      Value.HasShape and (Length(Value.Shape) = 4), 'YOLO graph input differs from the profile');
    NativeRequire((Value.Shape[0].Value = 1) and (Value.Shape[1].Value = 3) and
      (Value.Shape[2].Value = Profile.InputHeight) and (Value.Shape[3].Value = Profile.InputWidth),
      'YOLO graph input dimensions differ from the profile');
  end;
  NativeRequire(Count = 1, 'Expected one YOLO graph input');
  NativeRequire(Length(Graph.Outputs) = Length(Profile.Outputs),
    'YOLO graph output count differs from the profile');
  for I := 0 to High(Profile.Outputs) do
  begin
    Value := Graph.Outputs[I];
    NativeRequire((Value.Name = Profile.Outputs[I].Name) and Value.HasShape and
      (Length(Value.Shape) = Length(Profile.Outputs[I].Shape)),
      'YOLO graph output name or rank differs from the profile');
    NativeRequire(((Profile.Outputs[I].DataType = yttFloat32) and (Value.DataType = OnnxFloat)) or
      ((Profile.Outputs[I].DataType = yttUInt8) and (Value.DataType = OnnxUInt8)),
      'YOLO graph output dtype differs from the profile');
    for J := 0 to High(Value.Shape) do
      NativeRequire(Value.Shape[J].Value = Profile.Outputs[I].Shape[J],
        'YOLO graph output shape differs from the profile');
  end;
end;

procedure DeleteStage(const Stage: string);
const
  Files: array[0..6] of string = ('manifest.json', 'model.so', 'model.kernels.o',
    'model.vm.o', 'model.dll', 'model.lib', 'model.exp');
var
  Name: string;
begin
  if Stage = '' then Exit;
  for Name in Files do
    {$IFDEF FPC}SysUtils.{$ELSE}System.SysUtils.{$ENDIF}DeleteFile(
      {$IFDEF FPC}{$IFDEF UNIX}UTF8Encode(IncludeTrailingPathDelimiter(Stage) + Name)
      {$ELSE}IncludeTrailingPathDelimiter(Stage) + Name{$ENDIF}
      {$ELSE}IncludeTrailingPathDelimiter(Stage) + Name{$ENDIF});
  RemoveDir(Stage);
end;

procedure InstallStage(const Stage, Destination: string);
begin
  {$IFDEF UNIX}
  NativeRequire(RenameNoReplace(-100, PAnsiChar(UTF8Encode(Stage)), -100,
    PAnsiChar(UTF8Encode(Destination)), 1) = 0, 'Could not install the new YOLO bundle');
  {$ELSE}
  NativeRequire(MoveFileW(PWideChar(Stage), PWideChar(Destination)),
    'Could not install the new YOLO bundle');
  {$ENDIF}
end;

function ConvertYoloBundle(const RequestFile: string; const ValidateOnly: Boolean): TJSONObject;
const
  LinuxTarget = '{"kind":"llvm","mtriple":"x86_64-pc-linux-gnu","mcpu":"haswell"}';
var
  Request, Catalog, Compiler, Runtime, Cuda, ProfileJSON, Manifest, Artifact, Provenance, RuntimePin: TJSONObject;
  Profiles: TYoloProfiles;
  Profile: TYoloProfile;
  Entries: TJSONArray;
  SourceStream, RecipeStream: TStream;
  RuntimeFiles: TArray<INativeFileLease>;
  ArtifactLease, CatalogLease: INativeFileLease;
  Recipe, Graph: TOnnxModel;
  Kit: TWindowsTvmKit;
  Base, SourcePath, RecipePath, Destination, ProfileID, Device, Architecture, Nvrtc, TargetOS: string;
  RuntimeDirectory, RuntimeSHA, Stage, LibraryFile, ModelPath, KernelFile, VMFile: string;
  RuntimeBytes: Int64;
  I, Selected: Integer;
  ID: TGUID;
begin
  Result := nil;
  Request := nil;
  Catalog := nil;
  Manifest := nil;
  SourceStream := nil;
  RecipeStream := nil;
  Recipe := nil;
  Graph := nil;
  Stage := '';
  try
    try
    Base := ExtractFileDir(ExpandFileName(RequestFile));
    Request := ReadJSONFile(RequestFile);
    NativeRequire(NativeText(Request, 'schema') = 'crosspascal.yolo-convert.v1',
      'Unsupported YOLO conversion request');
    TargetOS := NativeHostOS;
    if Request.GetValue('target_os') <> nil then TargetOS := NativeText(Request, 'target_os');
    NativeRequire((TargetOS = 'windows') or (TargetOS = 'linux'), 'Unsupported YOLO target OS');
    {$IFNDEF UNIX}
    NativeRequire(TargetOS = 'windows', 'This compiler host builds Windows YOLO libraries');
    {$ENDIF}
    if (Request.GetValue('catalog_sha256') <> nil) or (Request.GetValue('catalog_bytes') <> nil) then
    begin
      SourcePath := RequestPath(Base, Request, 'catalog');
      CatalogLease := HoldNativeFile(ExtractFileDir(SourcePath), ExtractFileName(SourcePath),
        NativeText(Request, 'catalog_sha256'), NativeInteger(Request, 'catalog_bytes'));
      Catalog := ReadNativeObject(CatalogLease, 2 * 1024 * 1024);
      NativeUniqueKeys(Catalog);
    end
    else
      Catalog := ReadJSONFile(RequestPath(Base, Request, 'catalog'));
    Profiles := ParseYoloProfiles(Catalog.ToJSON);
    ProfileID := NativeText(Request, 'profile');
    Selected := -1;
    for I := 0 to High(Profiles) do
      if Profiles[I].ID = ProfileID then Selected := I;
    NativeRequire(Selected >= 0, 'Unknown YOLO profile: ' + ProfileID);
    Profile := Profiles[Selected];
    Entries := NativeArray(NativeField(Catalog, 'profiles'));
    ProfileJSON := NativeObject(Entries.Items[Selected]);
    SourcePath := RequestPath(Base, Request, 'checkpoint');
    RecipePath := RequestPath(Base, Request, 'recipe');
    Destination := ExcludeTrailingPathDelimiter(RequestPath(Base, Request, 'destination'));
    NativeRequire((not DirectoryExists(Destination)) and (not FileExists(Destination)) and DirectoryExists(ExtractFileDir(Destination)),
      'Choose a new YOLO bundle directory in an existing parent');
    NativeRequire(SameText(ExtractFileExt(SourcePath), '.pt') and SameText(ExtractFileExt(RecipePath), '.onnx'),
      'Expected an original YOLO .pt checkpoint and its graph recipe');
    SourceStream := OpenRead(SourcePath);
    RecipeStream := OpenRead(RecipePath);
    NativeRequire(SHA256Stream(SourceStream) = Profile.SourceSHA256,
      'YOLO checkpoint does not match the selected profile');
    NativeRequire(SHA256Stream(RecipeStream) = Profile.RecipeSHA256,
      'YOLO graph recipe does not match the selected profile');
    Device := NativeText(Request, 'device');
    NativeRequire((Device = 'cpu') or (Device = 'cuda'), 'Unsupported YOLO conversion device');
    Architecture := '';
    Nvrtc := '';
    if Device = 'cuda' then
    begin
      Cuda := NativeObject(NativeField(Request, 'cuda'));
      Architecture := NativeText(Cuda, 'architecture');
      NativeRequire((Architecture = 'sm_75') or (Architecture = 'sm_86') or
        (Architecture = 'sm_89') or (Architecture = 'sm_120'), 'Unsupported YOLO CUDA target');
      Nvrtc := RequestPath(Base, Cuda, 'nvrtc');
      NativeRequire(FileExists(Nvrtc), 'YOLO NVRTC library does not exist');
    end
    else
      NativeRequire(Request.GetValue('cuda') = nil, 'CPU conversion does not use CUDA options');
    Compiler := NativeObject(NativeField(Request, 'compiler'));
    Kit.CompilerDirectory := RequestPath(Base, Compiler, 'directory');
    Kit.Linker := RequestPath(Base, Compiler, 'linker');
    Kit.FfiLibrary := RequestPath(Base, Compiler, 'ffi_library');
    if TargetOS = 'windows' then
    begin
    Kit.MsvcLibraries := RequestPath(Base, Compiler, 'msvc_libraries');
    Kit.UcrtLibraries := RequestPath(Base, Compiler, 'ucrt_libraries');
    Kit.WindowsLibraries := RequestPath(Base, Compiler, 'windows_libraries');
      NativeRequire(DirectoryExists(Kit.MsvcLibraries) and DirectoryExists(Kit.UcrtLibraries) and
        DirectoryExists(Kit.WindowsLibraries), 'The Windows YOLO link libraries are incomplete');
    end;
    NativeRequire(DirectoryExists(Kit.CompilerDirectory) and FileExists(Kit.Linker) and
      FileExists(Kit.FfiLibrary), 'The YOLO compiler kit is incomplete');
    Runtime := NativeObject(NativeField(Request, 'runtime'));
    RuntimeDirectory := RequestPath(Base, Runtime, 'directory');
    RuntimeSHA := NativeText(Runtime, 'manifest_sha256');
    RuntimeBytes := NativeInteger(Runtime, 'manifest_bytes');
    RuntimeFiles := CheckRuntime(RuntimeDirectory, RuntimeSHA, TargetOS, RuntimeBytes, Device = 'cuda');
    Recipe := TOnnxModel.Create(RecipePath);
    Graph := BindYoloCheckpoint(Recipe, SourcePath);
    CheckGraph(Graph, Profile);
    Result := TJSONObject.Create;
    Result.AddPair('profile', ProfileID);
    Result.AddPair('profile_identity', YoloProfileIdentity(Profile));
    Result.AddPair('device', Device);
    Result.AddPair('target_os', TargetOS);
    if Architecture <> '' then Result.AddPair('cuda_arch', Architecture);
    if ValidateOnly then
    begin
      Result.AddPair('status', 'YOLO_CHECKPOINT_AND_CONTRACT_VALIDATED');
      Exit;
    end;
    NativeRequire(CreateGUID(ID) = 0, 'Could not create a YOLO staging name');
    Stage := Destination + '.building-' + GUIDToString(ID);
    NativeRequire(CreateDir(Stage), 'Could not create the YOLO build directory');
    if TargetOS = 'windows' then
      LibraryFile := 'model.dll'
    else
      LibraryFile := 'model.so';
    ModelPath := NativeMember(Stage, LibraryFile);
    if TargetOS = 'windows' then
      CompileWindowsTvmOnnxLibrary(Graph, Kit, ModelPath, Nvrtc, Architecture)
    else begin
    {$IFDEF UNIX}
    KernelFile := NativeMember(Stage, 'model.kernels.o');
    VMFile := NativeMember(Stage, 'model.vm.o');
    if Architecture = '' then
      CompileCpuOnnx(Graph, Kit.CompilerDirectory, LinuxTarget, KernelFile, VMFile)
    else
      CompileCudaOnnx(Graph, Kit.CompilerDirectory, Nvrtc, Architecture, LinuxTarget, KernelFile, VMFile);
    LinkLinuxTvmLibrary(Kit.Linker, ModelPath, Kit.FfiLibrary, [KernelFile, VMFile]);
    NativeRequire(DeleteFile(UTF8Encode(KernelFile)) and DeleteFile(UTF8Encode(VMFile)),
      'Could not remove the temporary YOLO objects');
    {$ENDIF}
    end;
    Artifact := FilePin(ModelPath);
    Manifest := TJSONObject.Create;
    Manifest.AddPair('schema', 'crosspascal.yolo-bundle.v1');
    Manifest.AddPair('profile', ParseCrossJSON(ProfileJSON.ToJSON));
    Manifest.AddPair('profile_identity', YoloProfileIdentity(Profile));
    Manifest.AddPair('artifact', Artifact);
    Artifact.AddPair('file', LibraryFile);
    Artifact.AddPair('os', TargetOS);
    Artifact.AddPair('arch', 'x86_64');
    Artifact.AddPair('entry', 'main');
    Artifact.AddPair('tvm', '0.26.0');
    Artifact.AddPair('ffi_abi', '0.1.14');
    Artifact.AddPair('cpu_isa', 'avx2_fma');
    Artifact.AddPair('device', Device);
    if Architecture <> '' then Artifact.AddPair('cuda_arch', Architecture);
    ArtifactLease := HoldNativeFile(Stage, LibraryFile, NativeText(Artifact, 'sha256'),
      NativeInteger(Artifact, 'bytes'));
    RequireNativeX64LibraryForOS(ArtifactLease, TargetOS);
    ArtifactLease := nil;
    Provenance := TJSONObject.Create;
    Manifest.AddPair('provenance', Provenance);
    Provenance.AddPair('source_sha256', Profile.SourceSHA256);
    Provenance.AddPair('graph_recipe_sha256', Profile.RecipeSHA256);
    RuntimePin := TJSONObject.Create;
    Manifest.AddPair('runtime', RuntimePin);
    RuntimePin.AddPair('manifest_sha256', RuntimeSHA);
    RuntimePin.AddPair('manifest_bytes', JSONInteger(RuntimeBytes));
    WriteJSONFile(NativeMember(Stage, 'manifest.json'), Manifest);
    Result.AddPair('manifest', FilePin(NativeMember(Stage, 'manifest.json')));
    Result.AddPair('destination', Destination);
    Result.AddPair('status', 'NATIVE_YOLO_BUNDLE_CREATED');
    InstallStage(Stage, Destination);
    Stage := '';
  except
    Result.Free;
    Result := nil;
    raise;
    end;
  finally
    ArtifactLease := nil;
    CatalogLease := nil;
    DeleteStage(Stage);
    Graph.Free; Recipe.Free; RecipeStream.Free; SourceStream.Free;
    Manifest.Free; Catalog.Free; Request.Free;
  end;
end;

end.
