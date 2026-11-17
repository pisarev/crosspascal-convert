{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35Package                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35Package;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Qwen35Source, QwenCuda;

function InspectQwen35TextSource(const Request: TJSONObject;
  out Config: TQwen35Config): TJSONObject;
procedure PackageQwen35Text(const Request: TJSONObject; const Progress: TQwenProgress = nil);

implementation

uses
  {$IFDEF FPC}Windows, SysUtils, Classes, Generics.Collections
  {$ELSE}Winapi.Windows, System.SysUtils, System.Classes, System.Generics.Collections{$ENDIF},
  CrossPascal.GGUF, CrossPascal.NativeBundle, CrossPascal.Onnx, CrossPascal.Qwen35Manifest,
  CrossPascal.Qwen35Bundle, CrossPascal.Qwen35PackedCompiler, CrossPascal.Qwen35Tokenizer,
  CrossPascal.Qwen35VisionBundle, CrossPascal.Qwen35VisionGraph, CrossPascal.SafeTensors,
  CrossPascal.TVMCompiler, CrossPascal.SHA256, QwenTokenizer, TVM.CpuGraph;

function QwenDiskFreeSpace(const DirectoryName: PWideChar; Available, Total, Free: Pointer): BOOL;
  stdcall; external 'kernel32.dll' name 'GetDiskFreeSpaceExW';

function QwenCreateHardLink(const FileName, ExistingFileName: PWideChar;
  SecurityAttributes: Pointer): BOOL;
  stdcall; external 'kernel32.dll' name 'CreateHardLinkW';

const
  QwenCompilerReserve = Int64(1024) * 1024 * 1024;
  QwenVisualPrefix = 'model.visual.';
  QwenVisionDefaultSide = 512;
  QwenVisionMaxSide = 4096;
  QwenVisionFields: array[0..5] of string = ('vision_config', 'vision_source_repository',
    'vision_source_revision', 'vision_license_file', 'vision_height', 'vision_width');
  QwenVisionNames: array[0..2] of string = ('vision.dll', 'vision-parameters.bin', 'vision.json');

type
  { Optional image encoder request. Root is the parsed vision_config, owned by the caller. }
  TQwen35VisionRequest = record
    Root: TJSONObject;
    Shard: string;
    ConfigFile: string;
    Repository: string;
    Revision: string;
    LicenseFile: string;
    Height: Integer;
    Width: Integer;
    Tensors: Integer;
    WeightBytes: Int64;
    ShardBytes: Int64;
  end;

function QwenVisionSide(const Request: TJSONObject; const Field: string): Integer;
var
  Value: TJSONValue;
  Side: Int64;
begin
  Value := Request.GetValue(Field);
  if Value = nil then Exit(QwenVisionDefaultSide);
  Side := ModelInt(Value, Field);
  RequireCpu((Side > 0) and (Side <= QwenVisionMaxSide), Field + ' must be from 1 to 4096 pixels');
  Result := Integer(Side);
end;

function QwenRequestFile(const Request: TJSONObject; const Field, MessageText: string): string;
begin
  Result := ModelText(Request.GetValue(Field));
  RequireCpu((Result <> '') and FileExists(Result), MessageText);
  Result := ExpandFileName(Result);
end;

{ Validates the optional vision request against the text source by metadata only:
  no compilation and no file writes. Returns False with Root = nil when vision_source
  is absent; then every other vision field must be absent too. On success the caller
  owns Vision.Root. The source shard is open only inside this call. }
function InspectQwen35Vision(const Request: TJSONObject; const Config: TQwen35Config;
  var Vision: TQwen35VisionRequest): Boolean;
var
  Parsed: TJSONValue;
  TextConfig: TJSONObject;
  VisionTextConfig: TQwen35Config;
  Model: TOnnxModel;
  Shard: TSafeTensorFile;
  Used: TDictionary<string, Boolean>;
  GraphInput: TOnnxValue;
  Info: TSafeTensorInfo;
  Storage: TSafeTensorStorage;
  Name: string;
  I, J: Integer;
begin
  Vision.Root := nil;
  Vision.Tensors := 0;
  Vision.WeightBytes := 0;
  Vision.ShardBytes := 0;
  Result := Request.GetValue('vision_source') <> nil;
  if not Result then
  begin
    for Name in QwenVisionFields do
      RequireCpu(Request.GetValue(Name) = nil,
        Name + ' requires vision_source: the original safetensors shard with the model.visual.* tensors');
    Exit;
  end;
  Vision.Shard := QwenRequestFile(Request, 'vision_source',
    'vision_source must name the original safetensors shard with the model.visual.* tensors');
  Vision.ConfigFile := QwenRequestFile(Request, 'vision_config',
    'vision_config must name the original model config with vision_config and text_config');
  Vision.Repository := ModelText(Request.GetValue('vision_source_repository'));
  Vision.Revision := ModelText(Request.GetValue('vision_source_revision'));
  RequireCpu((Vision.Repository <> '') and (Length(Vision.Repository) <= 512) and
    ((Length(Vision.Revision) = 40) or (Length(Vision.Revision) = 64)),
    'Supply vision_source_repository and a pinned 40- or 64-character vision_source_revision');
  for I := 1 to Length(Vision.Revision) do
    RequireCpu(CharInSet(Vision.Revision[I], ['0'..'9', 'a'..'f', 'A'..'F']),
      'Invalid vision source revision');
  Vision.Revision := LowerCase(Vision.Revision);
  Vision.LicenseFile := QwenRequestFile(Request, 'vision_license_file',
    'The vision source license is required');
  Vision.Height := QwenVisionSide(Request, 'vision_height');
  Vision.Width := QwenVisionSide(Request, 'vision_width');
  Model := nil;
  Shard := nil;
  Used := nil;
  Parsed := ReadModelJSON(Vision.ConfigFile);
  try
    try
      TextConfig := ModelObject(ModelObject(Parsed, 'vision config').GetValue('text_config'), 'text_config');
      RequireCpu(ModelInt(TextConfig.GetValue('hidden_size'), 'text_config.hidden_size') = Config.Hidden,
        'The vision config text hidden_size differs from the text source');
      VisionTextConfig := ReadQwen35Config(TJSONObject(Parsed));
      RequireCpu((VisionTextConfig.Layers = Config.Layers) and
        (VisionTextConfig.Heads = Config.Heads) and (VisionTextConfig.KVHeads = Config.KVHeads) and
        (VisionTextConfig.HeadSize = Config.HeadSize) and
        (VisionTextConfig.RotaryDimensions = Config.RotaryDimensions) and
        (VisionTextConfig.LinearHeads = Config.LinearHeads) and
        (VisionTextConfig.LinearKVHeads = Config.LinearKVHeads) and
        (VisionTextConfig.LinearKeySize = Config.LinearKeySize) and
        (VisionTextConfig.LinearValueSize = Config.LinearValueSize),
        'The vision config text topology differs from the text source');
      for I := 0 to 2 do
        RequireCpu(VisionTextConfig.MRopeSections[I] = Config.MRopeSections[I],
          'The vision config MRoPE sections differ from the text source');
      for I := 0 to Config.Layers - 1 do
        RequireCpu(VisionTextConfig.Attention[I] = Config.Attention[I],
          'The vision config attention layout differs from the text source');
      Model := BuildQwen35VisionGraph(TJSONObject(Parsed), Vision.Height, Vision.Width);
      RequireCpu((Length(Model.Inputs) >= 2) and (Model.Inputs[0].Name = 'patches'),
        'The Qwen vision graph has no weight inputs');
      Shard := TSafeTensorFile.Create(Vision.Shard);
      Used := TDictionary<string, Boolean>.Create;
      for I := 1 to High(Model.Inputs) do
      begin
        GraphInput := Model.Inputs[I];
        Name := GraphInput.Name;
        RequireCpu(GraphInput.HasShape and not Used.ContainsKey(Name),
          'Invalid Qwen vision graph input: ' + Name);
        Used.Add(Name, True);
        Info := Shard.TensorInfo(Name);
        RequireCpu((Info.DType = 'BF16') and (Length(Info.Shape) = Length(GraphInput.Shape)),
          'Vision source tensor is not BF16 with the graph rank: ' + Name);
        for J := 0 to High(Info.Shape) do
          RequireCpu((GraphInput.Shape[J].Symbol = '') and (GraphInput.Shape[J].Value > 0) and (Info.Shape[J] = GraphInput.Shape[J].Value),
            'Vision source tensor shape differs from the graph: ' + Name);
        Storage := Shard.TensorStorage(Name);
        RequireCpu(Vision.WeightBytes <= High(Int64) - Storage.ByteCount,
          'Vision source byte total overflow');
        Inc(Vision.WeightBytes, Storage.ByteCount);
        Vision.ShardBytes := Storage.FileSize;
      end;
      for Name in Shard.TensorNames do
        if Copy(Name, 1, Length(QwenVisualPrefix)) = QwenVisualPrefix then
        begin
          RequireCpu(Used.ContainsKey(Name), 'Source visual tensor is not used by the vision graph: ' + Name);
          Inc(Vision.Tensors);
        end;
      RequireCpu(Vision.Tensors = Used.Count, 'The vision graph and the source visual tensor set differ');
      Vision.Root := TJSONObject(Parsed);
    except
      Parsed.Free;
      raise;
    end;
  finally
    Used.Free;
    Shard.Free;
    Model.Free;
  end;
end;

function InspectQwen35Model(const Request: TJSONObject; out Config: TQwen35Config;
  out Vision: TQwen35VisionRequest): TJSONObject;
var
  Dimensions, Details: TJSONObject;
  I, FullCount: Integer;
begin
  Vision.Root := nil;
  Result := BuildQwen35TextManifest(ModelText(Request.GetValue('source')),
    ModelText(Request.GetValue('ggml')), ModelInt(Request.GetValue('context'), 'context'),
    ModelInt(Request.GetValue('head_rows'), 'head rows'), Config,
    Request.GetValue('vision_source') <> nil);
  try
    Dimensions := ModelObject(Result.GetValue('config'), 'dimensions');
    Dimensions.AddPair('attention_heads', JSONInteger(Config.Heads));
    Dimensions.AddPair('kv_heads', JSONInteger(Config.KVHeads));
    Dimensions.AddPair('head_size', JSONInteger(Config.HeadSize));
    Dimensions.AddPair('rotary_dimensions', JSONInteger(Config.RotaryDimensions));
    Dimensions.AddPair('linear_heads', JSONInteger(Config.LinearHeads));
    Dimensions.AddPair('linear_kv_heads', JSONInteger(Config.LinearKVHeads));
    Dimensions.AddPair('linear_key_size', JSONInteger(Config.LinearKeySize));
    Dimensions.AddPair('linear_value_size', JSONInteger(Config.LinearValueSize));
    Dimensions.AddPair('conv_kernel', JSONInteger(Config.ConvKernel));
    FullCount := 0;
    for I := 0 to High(Config.Attention) do
      if Config.Attention[I] = qaFullAttention then Inc(FullCount);
    Result.AddPair('full_attention_layers', JSONInteger(FullCount));
    Result.AddPair('linear_attention_layers', JSONInteger(Config.Layers - FullCount));
    Result.AddPair('source_max_context', JSONInteger(Config.Context));
    Result.AddPair('mtp_layers_in_source', JSONInteger(Config.MtpLayers));
    if InspectQwen35Vision(Request, Config, Vision) then
    begin
      Result.AddPair('vision_requested', JSONBoolean(True));
      Result.AddPair('vision_supported', JSONBoolean(True));
      Details := TJSONObject.Create;
      Result.AddPair('vision_encoder', Details);
      Details.AddPair('image_height', JSONInteger(Vision.Height));
      Details.AddPair('image_width', JSONInteger(Vision.Width));
      Details.AddPair('weight_tensors', JSONInteger(Vision.Tensors));
      Details.AddPair('weight_bytes', JSONInteger(Vision.WeightBytes));
      Details.AddPair('weight_encoding', 'bf16');
      Details.AddPair('source_repository', Vision.Repository);
      Details.AddPair('source_revision', Vision.Revision);
    end
    else
      Result.AddPair('vision_supported', JSONBoolean(False));
    Result.AddPair('mtp_supported', JSONBoolean(False));
  except
    FreeAndNil(Vision.Root);
    Result.Free;
    raise;
  end;
end;

function InspectQwen35TextSource(const Request: TJSONObject;
  out Config: TQwen35Config): TJSONObject;
var
  Vision: TQwen35VisionRequest;
begin
  Result := InspectQwen35Model(Request, Config, Vision);
  FreeAndNil(Vision.Root);
end;

function FileDigest(const FileName: string; out Bytes: Int64): string;
var
  Handle: THandle;
  Stream: THandleStream;
begin
  Handle := CreateFileW(PWideChar(FileName), GENERIC_READ, FILE_SHARE_READ, nil,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL or FILE_FLAG_SEQUENTIAL_SCAN, 0);
  if Handle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  try
    Stream := THandleStream.Create(Handle);
    try
      Bytes := Stream.Size;
      Result := SHA256Stream(Stream);
    finally
      Stream.Free;
    end;
  finally
    CloseHandle(Handle);
  end;
end;

procedure PackageQwen35Text(const Request: TJSONObject; const Progress: TQwenProgress);
const
  GraphNames: array[0..2] of string = ('linear.dll', 'full.dll', 'head.dll');
  PrefillGraphNames: array[0..1] of string = ('prefill-linear.dll', 'prefill-full.dll');
  ReaderNames: array[0..3] of string = ('ggml-base.dll', 'libunwind.dll', 'libc++.dll', 'libomp.dll');
var
  Config: TQwen35Config;
  Manifest, Sources, Hashes, KitJSON, Licenses: TJSONObject;
  Kit: TWindowsTvmKit;
  Tokenizer: TQwenTokenizer;
  GGUF: TGGUFCheckpoint;
  Source, OutputFolder, Stage, Name, SourceName, SourceFolder, Target,
    Architecture, Nvrtc, ReaderFolder, Repository, Revision: string;
  I, EffectiveVocab: Integer;
  Bytes, RequiredBytes, CacheBytes: Int64;
  FreeBytes, TotalBytes: UInt64;
  LinkWeights, IsGGUF, PoolBuffers: Boolean;
  PoolRequest, VisionParsed: TJSONValue;
  ID: TGUID;
  Vision: TQwen35VisionRequest;
  VisionManifest, VisionSource, VisionFile: TJSONObject;
  VisionFiles, Modalities: TJSONArray;
  VisionStage, VisionDigest: string;

  procedure Notify(const Text: string);
  begin
    if Assigned(Progress) then Progress(Text);
  end;

  procedure AddArtifact(const Member: string; const ExpectedBytes: Int64 = -1);
  var
    Size: Int64;
    SHA: string;
  begin
    RequireCpu(Hashes.GetValue(Member) = nil, 'Duplicate bundle artifact: ' + Member);
    SHA := FileDigest(BundleMember(Stage, Member), Size);
    RequireCpu((ExpectedBytes < 0) or (Size = ExpectedBytes), 'Source file size changed: ' + Member);
    Hashes.AddPair(Member, SHA);
  end;

  procedure CopyArtifact(const FileName, Member: string);
  begin
    RequireCpu(CopyFileW(PWideChar(ExpandFileName(FileName)),
      PWideChar(BundleMember(Stage, Member)), True), 'Could not copy bundle member: ' + Member);
    AddArtifact(Member);
  end;

begin
  Source := ExpandFileName(ModelText(Request.GetValue('source')));
  OutputFolder := ExcludeTrailingPathDelimiter(ExpandFileName(ModelText(Request.GetValue('destination'))));
  RequireCpu((ModelText(Request.GetValue('destination')) <> '') and
    DirectoryExists(ExtractFileDir(OutputFolder)) and not FileExists(OutputFolder) and
    not DirectoryExists(OutputFolder), 'Choose a new bundle directory in an existing parent');
  SourceFolder := Source;
  IsGGUF := not DirectoryExists(Source);
  if IsGGUF then SourceFolder := ExtractFileDir(Source);
  RequireCpu(not SameText(SourceFolder, OutputFolder) and
    not SameText(Copy(OutputFolder, 1, Length(IncludeTrailingPathDelimiter(SourceFolder))),
    IncludeTrailingPathDelimiter(SourceFolder)), 'Bundle output must be outside the checkpoint');
  Target := ModelText(Request.GetValue('target'));
  RequireCpu((Target = 'cpu') or (Target = 'cuda'), 'Select CPU or CUDA');
  RequireCpu(IsGGUF or (Target = 'cuda'), 'Original BF16 conversion requires CUDA');
  CacheBytes := 0;
  if IsGGUF and (Target = 'cuda') then CacheBytes := Int64(2) * 1024 * 1024 * 1024;
  if Request.GetValue('cuda_packed_cache_bytes') <> nil then
    CacheBytes := ModelInt(Request.GetValue('cuda_packed_cache_bytes'), 'packed cache budget');
  RequireCpu((CacheBytes >= 0) and (CacheBytes <= Int64(64) * 1024 * 1024 * 1024),
    'The packed CUDA cache budget must be from 0 to 64 GiB');
  RequireCpu((CacheBytes = 0) or (IsGGUF and (Target = 'cuda')),
    'Packed CUDA caching requires a GGUF CUDA bundle');
  PoolRequest := Request.GetValue('cuda_decoded_buffer_pool');
  RequireCpu((PoolRequest = nil) or JSONIsBoolean(PoolRequest),
    'cuda_decoded_buffer_pool must be a JSON boolean');
  PoolBuffers := (PoolRequest <> nil) and JSONIsTrue(PoolRequest);
  RequireCpu(not PoolBuffers or (IsGGUF and (Target = 'cuda')),
    'Decoded CUDA buffer pooling requires a GGUF CUDA bundle');
  Repository := ModelText(Request.GetValue('source_repository'));
  Revision := ModelText(Request.GetValue('source_revision'));
  RequireCpu((Repository <> '') and (Length(Repository) <= 512) and ((Length(Revision) = 40) or (Length(Revision) = 64)),
    'Supply the original repository and pinned revision');
  for I := 1 to Length(Revision) do
    RequireCpu(CharInSet(Revision[I], ['0'..'9', 'a'..'f', 'A'..'F']), 'Invalid source revision');
  RequireCpu(FileExists(ModelText(Request.GetValue('license_file'))), 'The source model license is required');
  KitJSON := ModelObject(Request.GetValue('kit'), 'Windows TVM toolchain');
  Kit.CompilerDirectory := ModelText(KitJSON.GetValue('compiler'));
  Kit.Linker := ModelText(KitJSON.GetValue('linker'));
  Kit.FfiLibrary := ModelText(KitJSON.GetValue('ffi_library'));
  Kit.MsvcLibraries := ModelText(KitJSON.GetValue('msvc_libraries'));
  Kit.UcrtLibraries := ModelText(KitJSON.GetValue('ucrt_libraries'));
  Kit.WindowsLibraries := ModelText(KitJSON.GetValue('windows_libraries'));
  Nvrtc := '';
  Architecture := '';
  if Target = 'cuda' then
  begin
    Nvrtc := ModelText(KitJSON.GetValue('nvrtc_library'));
    Architecture := ModelText(Request.GetValue('cuda_arch'));
    RequireCpu((Nvrtc <> '') and (Copy(Architecture, 1, 3) = 'sm_'), 'Specify NVRTC and CUDA architecture');
  end;
  LinkWeights := JSONIsTrue(Request.GetValue('hardlink_weights'));
  Stage := '';
  VisionStage := '';
  Manifest := InspectQwen35Model(Request, Config, Vision);
  try
    Sources := ModelObject(Manifest.GetValue('source_files'), 'source files');
    RequiredBytes := QwenCompilerReserve;
    if Vision.Root <> nil then
    begin
      { Unchanged BF16 visual bytes are always copied; the second reserve covers the vision compile. }
      RequireCpu(Vision.WeightBytes <= High(Int64) - 2 * QwenCompilerReserve,
        'Vision source byte total overflow');
      Inc(RequiredBytes, Vision.WeightBytes + QwenCompilerReserve);
    end;
    if not LinkWeights then
      for I := 0 to Sources.Count - 1 do
      begin
        Bytes := ModelInt(ModelObject(Sources.GetValue(JSONObjectName(Sources, I)),
          'source file').GetValue('bytes'), 'source bytes');
        RequireCpu(Bytes <= High(Int64) - RequiredBytes, 'Source byte total overflow');
        Inc(RequiredBytes, Bytes);
      end;
    RequireCpu(QwenDiskFreeSpace(PWideChar(ExtractFileDir(OutputFolder)), @FreeBytes, @TotalBytes, nil),
      'Could not measure destination disk space');
    RequireCpu(FreeBytes >= UInt64(RequiredBytes),
      'Not enough disk space for original source files and build');
    RequireCpu(CreateGUID(ID) = 0, 'Could not name the Qwen build directory');
    Stage := OutputFolder + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(Stage), 'Could not create the Qwen build directory');
    Hashes := TJSONObject.Create;
    Manifest.AddPair('artifact_sha256', Hashes);
    Manifest.AddPair('source_repository', Repository);
    Manifest.AddPair('source_revision', LowerCase(Revision));
    Manifest.AddPair('experimental', JSONBoolean(True));
    Manifest.AddPair('tokenizer', 'tokenizer.json');
    if LinkWeights then
      Manifest.AddPair('source_materialization', 'development_hardlinks')
    else
      Manifest.AddPair('source_materialization', 'copied_original_files');
    if IsGGUF then
      Manifest.AddPair('backend', 'TVM ' + UpperCase(Target) + ' mixed GGUF streamed')
    else
      Manifest.AddPair('backend', 'TVM CUDA BF16 streamed');
    if IsGGUF and (Target = 'cuda') then
    begin
      Manifest.AddPair('cuda_packed_cache_bytes', JSONInteger(CacheBytes));
      if PoolRequest <> nil then
        Manifest.AddPair('cuda_decoded_buffer_pool', JSONBoolean(PoolBuffers));
    end;
    Name := ModelText(Request.GetValue('cuda_projection_schedule'));
    if (Name = '') and IsGGUF and (Target = 'cuda') then Name := 'row_reduce_128';
    if Name = 'legacy' then Name := '';
    if Name <> '' then
    begin
      RequireCpu(IsGGUF and (Target = 'cuda') and (Name = 'row_reduce_128'),
        'The row projection schedule requires a GGUF CUDA bundle');
      Manifest.AddPair('cuda_projection_schedule', Name);
    end;
    for I := 0 to Sources.Count - 1 do
    begin
      Name := JSONObjectName(Sources, I);
      SourceName := BundleMember(SourceFolder, Name);
      Bytes := ModelInt(ModelObject(Sources.GetValue(Name), 'source file').GetValue('bytes'), 'source bytes');
      Notify('Preserving and verifying original source file ' + Name + '...');
      if LinkWeights then
        RequireCpu(QwenCreateHardLink(PWideChar(BundleMember(Stage, Name)), PWideChar(SourceName), nil),
          'Could not hardlink original source on the same volume')
      else
        RequireCpu(CopyFileW(PWideChar(SourceName), PWideChar(BundleMember(Stage, Name)), True),
          'Could not copy original source: ' + Name);
      AddArtifact(Name, Bytes);
    end;
    CopyArtifact(ModelText(Request.GetValue('license_file')), 'LICENSE-model.txt');
    if IsGGUF then
    begin
      ReaderFolder := ExtractFileDir(ExpandFileName(ModelText(Request.GetValue('ggml'))));
      for Name in ReaderNames do CopyArtifact(BundleMember(ReaderFolder, Name), Name);
      Manifest.AddPair('tensor_decoder', 'ggml-base.dll');
      Licenses := ModelObject(Request.GetValue('reader_licenses'), 'tensor reader licenses');
      RequireCpu(Licenses.Count > 0, 'Tensor reader licenses are required');
      for I := 0 to Licenses.Count - 1 do
      begin
        Name := JSONObjectName(Licenses, I);
        RequireCpu(SameText(ExtractFileExt(Name), '.txt'), 'Use flat .txt names for reader licenses');
        CopyArtifact(ModelText(Licenses.GetValue(Name)), Name);
      end;
      Notify('Reconstructing the tokenizer from the original GGUF...');
      GGUF := TGGUFCheckpoint.Create(Source, ModelText(Request.GetValue('ggml')));
      try
        EffectiveVocab := ExportQwen35GGUFTokenizer(GGUF, BundleMember(Stage, 'tokenizer.json'));
      finally
        GGUF.Free;
      end;
    end
    else begin
      CopyArtifact(BundleMember(SourceFolder, 'config.json'), 'config.json');
      CopyArtifact(BundleMember(SourceFolder, 'tokenizer.json'), 'tokenizer.json');
      EffectiveVocab := 0;
    end;
    Tokenizer := TQwenTokenizer.Create;
    try
      Tokenizer.LoadByteLevelFile(BundleMember(Stage, 'tokenizer.json'));
      if IsGGUF then
        RequireCpu(Tokenizer.TokenCount = EffectiveVocab, 'Exported tokenizer vocabulary differs')
      else
        EffectiveVocab := Tokenizer.TokenCount;
      RequireCpu(
        (EffectiveVocab > 0) and (EffectiveVocab <= Config.Vocab) and (Tokenizer.SpecialId('<|im_end|>') >= 0) and (Tokenizer.SpecialId('<|endoftext|>') >= 0),
        'Unsupported Qwen tokenizer vocabulary or chat markers'
      );
    finally
      Tokenizer.Free;
    end;
    Manifest.AddPair('effective_vocab', JSONInteger(EffectiveVocab));
    if IsGGUF then AddArtifact('tokenizer.json');
    CompileQwen35TextGraphs(Manifest, Config, Stage, Kit, Nvrtc, Architecture, Progress);
    for Name in GraphNames do AddArtifact(Name);
    { Optional batched prefill libraries exist only when the compiler added
      prefill_graphs; older manifests stay valid without them. }
    if Manifest.GetValue('prefill_graphs') <> nil then
      for Name in PrefillGraphNames do AddArtifact(Name);
    if IsGGUF and (Target = 'cuda') then
    begin
      Notify('Compiling original-format tensor unpacking for ' + Architecture + '...');
      Name := 'packed-' + Architecture + '.cubin';
      CompileQwen35PackedDecoder(ModelText(Request.GetValue('packed_kernel_directory')),
        Nvrtc, Architecture, BundleMember(Stage, Name));
      AddArtifact(Name);
      Manifest.AddPair('packed_decoder', Name);
      CopyArtifact(BundleMember(ModelText(Request.GetValue('packed_kernel_directory')), 'LICENSE'),
        'LICENSE-packed-source.txt');
      CopyArtifact(BundleMember(ModelText(Request.GetValue('packed_kernel_directory')), 'PIN.json'),
        'packed-source.json');
    end;
    if Vision.Root <> nil then
    begin
      Notify('Building the image encoder from the original BF16 visual tensors');
      VisionStage := BundleMember(Stage, 'vision-stage');
      RequireCpu(not FileExists(VisionStage) and not DirectoryExists(VisionStage) and CreateDir(VisionStage),
        'Could not create the Qwen vision build directory');
      WriteQwen35VisionBundle(Vision.Root, Vision.Shard, VisionStage, Vision.Height, Vision.Width,
        Kit, Nvrtc, Architecture);
      for Name in QwenVisionNames do
      begin
        RequireCpu(FileExists(BundleMember(VisionStage, Name)), 'Qwen vision output is missing: ' + Name);
        RequireCpu(not FileExists(BundleMember(Stage, Name)) and not DirectoryExists(BundleMember(Stage, Name)),
          'Qwen vision output would replace a bundle member: ' + Name);
        RequireCpu(MoveFileW(PWideChar(BundleMember(VisionStage, Name)), PWideChar(BundleMember(Stage, Name))),
          'Could not move Qwen vision output: ' + Name);
      end;
      RequireCpu(RemoveDir(VisionStage), 'The Qwen vision build directory is not empty');
      VisionStage := '';
      for Name in QwenVisionNames do AddArtifact(Name);
      VisionParsed := ReadModelJSON(BundleMember(Stage, 'vision.json'));
      try
        VisionManifest := ModelObject(VisionParsed, 'vision manifest');
        RequireCpu(ModelText(VisionManifest.GetValue('schema')) = 'crosspascal-qwen35-vision-v1',
          'Unexpected Qwen vision manifest schema');
        RequireCpu((ModelText(VisionManifest.GetValue('device')) = Target) and
          (ModelText(VisionManifest.GetValue('cuda_arch')) = Architecture),
          'Qwen vision device differs from the bundle target');
        RequireCpu((ModelText(VisionManifest.GetValue('library')) = 'vision.dll') and
          (ModelText(VisionManifest.GetValue('weights_file')) = 'vision-parameters.bin'),
          'Unexpected Qwen vision file names');
        RequireCpu(ModelInt(VisionManifest.GetValue('source_file_size'), 'vision source size') = Vision.ShardBytes,
          'Qwen vision source shard size changed');
        VisionDigest := LowerCase(ModelText(VisionManifest.GetValue('source_sha256')));
        RequireCpu(Length(VisionDigest) = 64, 'Invalid Qwen vision source digest');
        for I := 1 to Length(VisionDigest) do
          RequireCpu(CharInSet(VisionDigest[I], ['0'..'9', 'a'..'f']), 'Invalid Qwen vision source digest');
        VisionFiles := ModelArray(VisionManifest.GetValue('files'), 'vision files');
        RequireCpu(VisionFiles.Count = 2, 'Unexpected Qwen vision file list');
        for I := 0 to VisionFiles.Count - 1 do
        begin
          VisionFile := ModelObject(VisionFiles.Items[I], 'vision file');
          Name := ModelText(VisionFile.GetValue('file'));
          RequireCpu(((Name = 'vision.dll') or (Name = 'vision-parameters.bin')) and
            SameText(ModelText(VisionFile.GetValue('sha256')), ModelText(Hashes.GetValue(Name))),
            'Qwen vision file hash differs from the bundle: ' + Name);
        end;
      finally
        VisionParsed.Free;
      end;
      CopyArtifact(Vision.ConfigFile, 'vision-config.json');
      CopyArtifact(Vision.LicenseFile, 'LICENSE-vision-model.txt');
      Manifest.AddPair('vision_manifest', 'vision.json');
      Modalities := TJSONArray.Create;
      Manifest.AddPair('modalities', Modalities);
      Modalities.AddElement(TJSONString.Create('text'));
      Modalities.AddElement(TJSONString.Create('image'));
      VisionSource := TJSONObject.Create;
      Manifest.AddPair('vision_source', VisionSource);
      VisionSource.AddPair('repository', Vision.Repository);
      VisionSource.AddPair('revision', Vision.Revision);
      VisionSource.AddPair('sha256', VisionDigest);
      VisionSource.AddPair('file_size', JSONInteger(Vision.ShardBytes));
      VisionSource.AddPair('config', 'vision-config.json');
      VisionSource.AddPair('license_file', 'LICENSE-vision-model.txt');
    end;
    WriteBundleJSON(BundleMember(Stage, 'model.json'), Manifest);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(OutputFolder)),
      'Could not install the completed Qwen bundle');
    Stage := '';
    Notify('Qwen text bundle created: ' + OutputFolder);
  finally
    Manifest.Free;
    Vision.Root.Free;
    if Stage <> '' then
    begin
      if VisionStage <> '' then DeleteBundleStage(VisionStage);
      DeleteBundleStage(Stage);
    end;
  end;
end;

end.
