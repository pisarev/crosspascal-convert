{ ************************************************************************** }
{                                                                            }
{ ConvertQwen2                                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertQwen2;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}Windows, SysUtils{$ELSE}Winapi.Windows, System.SysUtils{$ENDIF},
  CrossPascal.Json, CrossPascal.NativeBundle, CrossPascal.Onnx, CrossPascal.Qwen2Graph,
  CrossPascal.SafeTensors, CrossPascal.TVMCompiler, CrossPascal.TVMOnnx, TVM.CpuGraph;

const
  DecodeLibrary = 'qwen2-cuda.dll';
  PrefillLibrary = 'qwen2-prefill-cuda.dll';
  ParameterFile = 'weights.bin';
  MetadataFiles: array[0..4] of string = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
    'generation_config.json', 'LICENSE');

procedure Convert(const Source, Destination: string; const Context, Chunk: Integer;
  const NvrtcLibrary, Architecture: string; const Kit: TWindowsTvmKit);
var
  Data: TJSONValue;
  Config: TQwen2Config;
  Checkpoint: TSafeTensorCheckpoint;
  Decode, Prefill: TOnnxModel;
  Manifest, Hashes: TJSONObject;
  Layout: TJSONArray;
  Output, Stage, Name, SourcePath, DestPath: string;
  ID: TGUID;
begin
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu(DirectoryExists(ExtractFileDir(Output)) and not DirectoryExists(Output) and
    not FileExists(Output), 'Choose a new bundle directory in an existing parent');
  RequireCpu((Copy(Architecture, 1, 3) = 'sm_') and (Length(Architecture) >= 5),
    'Specify a CUDA compute architecture such as sm_89');
  Data := ReadModelJSON(BundleMember(Source, 'config.json'));
  Manifest := nil;
  Checkpoint := nil;
  Decode := nil;
  Prefill := nil;
  Stage := '';
  try
    Config := ReadQwen2Config(ModelObject(Data, 'Qwen2 configuration'));
    RequireCpu((Context >= 2) and (Context <= Config.MaxContext) and (Chunk >= 2) and (Chunk <= Context),
      'Invalid Qwen2 context or prefill chunk');
    Checkpoint := TSafeTensorCheckpoint.Create(Source);
    Decode := BuildQwen2Graph(Config, Checkpoint, Context, 1);
    Prefill := BuildQwen2Graph(Config, Checkpoint, Context, Chunk);
    RequireCpu(CreateGUID(ID) = 0, 'Could not name the Qwen2 build directory');
    Name := Output + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(Name), 'Could not create the Qwen2 build directory');
    Stage := Name;
    Manifest := TJSONObject.Create;
    Manifest.AddPair('schema', 'crosspascal.qwen2.text.v1');
    Manifest.AddPair('architecture', 'qwen2');
    Manifest.AddPair('target', 'cuda');
    Manifest.AddPair('backend', 'TVM CUDA FP32');
    Manifest.AddPair('cuda_arch', Architecture);
    Manifest.AddPair('library', DecodeLibrary);
    Manifest.AddPair('prefill_library', PrefillLibrary);
    Manifest.AddPair('context_window_size', JSONInteger(Context));
    Manifest.AddPair('prefill_chunk_size', JSONInteger(Chunk));
    Manifest.AddPair('cache_count', JSONInteger(2 * Config.Layers));
    Manifest.AddPair('cache_shape', BundleDimensions([1, Config.KVHeads, Context, Config.HeadSize]));
    Manifest.AddPair('vocab_size', JSONInteger(Config.Vocab));
    Layout := TJSONArray.Create;
    Manifest.AddPair('weights', Layout);
    Hashes := TJSONObject.Create;
    Manifest.AddPair('artifact_sha256', Hashes);
    for Name in MetadataFiles do
    begin
      SourcePath := BundleMember(Source, Name);
      if (Name <> 'config.json') and (Name <> 'tokenizer.json') and not FileExists(SourcePath) then
        Continue;
      DestPath := BundleMember(Stage, Name);
      if not CopyFileW(PWideChar(SourcePath), PWideChar(DestPath), True) then RaiseLastOSError;
      Hashes.AddPair(Name, BundleFileSHA256(DestPath));
    end;
    WriteLn('Writing Qwen2 weights from the source checkpoint');
    DestPath := BundleMember(Stage, ParameterFile);
    WriteGraphWeights(DestPath, Checkpoint, Decode, Prefill, 2 + 2 * Config.Layers, Layout);
    Hashes.AddPair(ParameterFile, BundleFileSHA256(DestPath));
    WriteLn('Compiling the Qwen2 decode graph for ', UTF8Encode(Architecture));
    DestPath := BundleMember(Stage, DecodeLibrary);
    CompileWindowsTvmOnnxLibrary(Decode, Kit, DestPath, NvrtcLibrary, Architecture);
    Hashes.AddPair(DecodeLibrary, BundleFileSHA256(DestPath));
    WriteLn('Compiling the Qwen2 prefill graph for ', UTF8Encode(Architecture));
    DestPath := BundleMember(Stage, PrefillLibrary);
    CompileWindowsTvmOnnxLibrary(Prefill, Kit, DestPath, NvrtcLibrary, Architecture);
    Hashes.AddPair(PrefillLibrary, BundleFileSHA256(DestPath));
    WriteBundleJSON(BundleMember(Stage, 'model.json'), Manifest);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(Output)), 'Could not install the Qwen2 bundle');
    Stage := '';
    WriteLn(UTF8Encode(Output));
  finally
    Prefill.Free;
    Decode.Free;
    Checkpoint.Free;
    Manifest.Free;
    Data.Free;
    if Stage <> '' then DeleteBundleStage(Stage);
  end;
end;

var
  Kit: TWindowsTvmKit;

begin
  if ParamCount <> 12 then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertQwen2 checkpoint-directory destination context prefill-chunk');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib nvrtc.dll sm_XX');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(5);
    Kit.Linker := ParamStr(6);
    Kit.FfiLibrary := ExpandFileName(ParamStr(7));
    Kit.MsvcLibraries := ParamStr(8);
    Kit.UcrtLibraries := ParamStr(9);
    Kit.WindowsLibraries := ParamStr(10);
    Convert(ParamStr(1), ParamStr(2), StrToInt(ParamStr(3)), StrToInt(ParamStr(4)), ParamStr(11),
      ParamStr(12), Kit);
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
