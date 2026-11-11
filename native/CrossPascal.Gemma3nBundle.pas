{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nBundle                                                  }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.TVMCompiler;

procedure ConvertGemma3nCpuBundle(const Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary: string;
  const Kit: TWindowsTvmKit; const GenerateTextGraphs: Boolean = True; const ContextOverride: Integer = 0;
  const ChunkOverride: Integer = 0);
procedure ConvertGemma3nCudaBundle(const Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary, NvrtcLibrary, Architecture: string;
  const Kit: TWindowsTvmKit; const ContextOverride: Integer = 0);
procedure AddGemma3nVision(const TextBundle, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary, NvrtcLibrary: string;
  const Kit: TWindowsTvmKit);

implementation

uses
  {$IFDEF FPC}Classes, Math, SysUtils, Windows{$ELSE}System.Classes, System.Math, System.SysUtils, Winapi.Windows{$ENDIF},
  CrossPascal.Gemma3nSource, CrossPascal.Gemma3nGraph, CrossPascal.Gemma3nVisionSource,
  CrossPascal.GGUF, CrossPascal.Json, CrossPascal.Onnx, CrossPascal.RowQuantization,
  CrossPascal.SentencePiece, CrossPascal.SHA256, CrossPascal.TVMOnnx, TVM.CpuGraph;

const
  TextOnnx = 'text.onnx';
  PrefillOnnx = 'prefill.onnx';
  VisionOnnx = 'vision.onnx';
  TextLibrary = 'gemma3n-cpu.dll';
  PrefillLibrary = 'gemma3n-prefill-cpu.dll';
  VisionLibrary = 'gemma3n-vision-cpu.dll';
  CudaTextLibrary = 'gemma3n-cuda.dll';
  CudaVisionLibrary = 'gemma3n-vision-cuda.dll';
  WindowsLLVM = '{"kind":"llvm","mtriple":"x86_64-pc-windows-msvc","mcpu":"haswell"}';
  VisionFiles: array[0..5] of string = (VisionOnnx, 'vision-parameters.json',
    'vision-params.bin', 'vision-hard.npy', 'vision-hard-norm.npy', 'vision-projection.npy');

function FileHash(const Path: string): string;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    Result := SHA256Stream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure SaveJSON(const Path: string; const Value: TJSONObject);
var
  Stream: TFileStream;
  Text: UTF8String;
begin
  Text := UTF8Encode(Value.ToJSON + sLineBreak);
  Stream := TFileStream.Create(Path, fmCreate);
  try
    Stream.WriteBuffer(Pointer(Text)^, Length(Text));
  finally
    Stream.Free;
  end;
end;

function Dimensions(const Shape: array of Int64): TJSONArray;
var
  Dim: Int64;
begin
  Result := TJSONArray.Create;
  for Dim in Shape do Result.AddElement(JSONInteger(Dim));
end;

function FloatValues(const Values: array of Single): TJSONArray;
var
  Value: Single;
begin
  Result := TJSONArray.Create;
  for Value in Values do Result.AddElement(JSONFloat(Value));
end;

procedure RequireValue(const Value: TOnnxValue; const Name: string; const DataType: Integer;
  const Shape: array of Int64);
var
  I: Integer;
begin
  RequireCpu((Value.Name = Name) and Value.HasShape and (Value.DataType = DataType) and
    (Length(Value.Shape) = Length(Shape)), 'Invalid Gemma 3n graph binding: ' + Name);
  for I := 0 to High(Shape) do
    RequireCpu(Value.Shape[I].Value = Shape[I], 'Gemma 3n graph dimensions differ: ' + Name);
end;

procedure AddGpuBinding(const Bindings: TJSONArray; const Source, Kind: string);
var
  Row: TJSONObject;
begin
  Row := TJSONObject.Create;
  Bindings.AddElement(Row);
  Row.AddPair('source', Source);
  Row.AddPair('kind', Kind);
end;

procedure QuantizeWeight(const Name, Folder: string; const Shape: TArray<Int64>;
  const Binding, Spec, Destination, Hashes: TJSONObject);
var
  Data: TMappedModelFile;
  Quantized, Scales: TFileStream;
  Slice: TJSONValue;
  Offset, Bytes: Int64;
  Q8File, ScaleFile, Number: string;
  Entry: TJSONObject;
begin
  RequireCpu((Length(Shape) = 2) and (Shape[0] <= High(Integer)) and (Shape[1] <= High(Integer)),
    'Invalid Gemma 3n quantization shape: ' + Name);
  Bytes := TensorElements(Shape) * SizeOf(Single);
  Offset := 0;
  Slice := Binding.GetValue('slice');
  if not (Slice is TJSONNull) then Offset := ModelInt(Slice, 'weight slice') * Bytes;
  Number := Format('%.4d.bin', [Destination.Count]);
  Q8File := 'gpu-q8-' + Number;
  ScaleFile := 'gpu-scale-' + Number;
  Data := TMappedModelFile.Create(BundleMember(Folder, ModelText(Spec.GetValue('file'))));
  Quantized := nil;
  Scales := nil;
  try
    Quantized := TFileStream.Create(BundleMember(Folder, Q8File), fmCreate);
    Scales := TFileStream.Create(BundleMember(Folder, ScaleFile), fmCreate);
    WriteSymmetricInt8Rows(Data.AtOffset(Offset, Bytes), Shape[0], Shape[1], Quantized, Scales);
    Quantized.Position := 0;
    Scales.Position := 0;
    Hashes.AddPair(Q8File, SHA256Stream(Quantized));
    Hashes.AddPair(ScaleFile, SHA256Stream(Scales));
  finally
    Scales.Free;
    Quantized.Free;
    Data.Free;
  end;
  Entry := TJSONObject.Create;
  Destination.AddPair(Name, Entry);
  Entry.AddPair('q8', Q8File);
  Entry.AddPair('scale', ScaleFile);
  Entry.AddPair('shape', Dimensions(Shape));
end;

procedure BindTextGraph(const Graph: TOnnxModel; const Config: TGemma3nConfig;
  const Context, Chunk: Integer; const Folder: string; const RecordData, Manifest: TJSONObject;
  const Gpu: Boolean = False; const ArtifactHashes: TJSONObject = nil);
const
  ClipSuffix = '.altup.correction_coefs.weight';
  CoefficientLimit = 120;
var
  Inputs, RuntimeInputs, GpuBindings: TJSONArray;
  Weights, Bindings, Params, Binding, Spec, Coefficient, Quantization, QuantizedWeights: TJSONObject;
  InputValue: TOnnxValue;
  CacheShape, Shape: TArray<Int64>;
  CacheCount, I, J: Integer;
  Original, Tensor, Name, FileName, Kind: string;
  Values: TArray<Single>;
  Stream: TFileStream;
  Slice: TJSONValue;
begin
  CacheCount := 2 * (Config.Layers - Config.SharedLayers);
  CacheShape := [1, Config.KVHeads, Context, Config.HeadSize];
  RequireCpu((Length(Graph.Inputs) > CacheCount + 3) and (Length(Graph.Outputs) = CacheCount + 1),
    'Gemma 3n graph cache count differs from the checkpoint');
  RequireValue(Graph.Inputs[0], 'embedding', 1, [1, Chunk, Config.Hidden]);
  RequireValue(Graph.Inputs[1], 'per_layer_embedding', 1,
    [1, Chunk, Int64(Config.Layers) * Config.LayerInput]);
  RequireValue(Graph.Inputs[2], 'position', 7, [Chunk]);
  RequireValue(Graph.Outputs[0], 'logits', 1, [1, Config.Vocab]);
  for I := 0 to CacheCount - 1 do
  begin
    RequireValue(Graph.Inputs[3 + I], 'cache_' + IntToStr(I), 1, CacheShape);
    RequireValue(Graph.Outputs[1 + I], 'updated_' + IntToStr(I), 1, CacheShape);
  end;
  Inputs := TJSONArray.Create;
  Manifest.AddPair('input_names', Inputs);
  RuntimeInputs := TJSONArray.Create;
  Manifest.AddPair('runtime_inputs', RuntimeInputs);
  Params := TJSONObject.Create;
  Manifest.AddPair('parameters', Params);
  GpuBindings := nil;
  QuantizedWeights := nil;
  if Gpu then
  begin
    GpuBindings := TJSONArray.Create;
    Manifest.AddPair('gpu_bindings', GpuBindings);
    Quantization := TJSONObject.Create;
    Manifest.AddPair('quantization', Quantization);
    Quantization.AddPair('scheme', 'symmetric-int8-per-row');
    Quantization.AddPair('activation_dtype', 'float32');
    QuantizedWeights := TJSONObject.Create;
    Quantization.AddPair('weights', QuantizedWeights);
  end;
  Weights := ModelObject(RecordData.GetValue('weights'), 'Gemma 3n weights');
  Bindings := ModelObject(RecordData.GetValue('parameters'), 'Gemma 3n checkpoint bindings');
  for I := 0 to High(Graph.Inputs) do
  begin
    InputValue := Graph.Inputs[I];
    Name := InputValue.Name;
    Inputs.AddElement(TJSONString.Create(Name));
    if I < CacheCount + 3 then
    begin
      RuntimeInputs.AddElement(TJSONString.Create(Name));
      if Gpu then AddGpuBinding(GpuBindings, Name, 'original');
      Continue;
    end;
    RequireCpu(Copy(Name, 1, 5) = 'text.', 'Unexpected Gemma 3n graph parameter: ' + Name);
    Original := Copy(Name, 6, MaxInt);
    Kind := 'original';
    if Gpu and (Bindings.GetValue(Original) = nil) then
    begin
      if Copy(Original, Length(Original) - 2, 3) = '_q8' then
      begin
        SetLength(Original, Length(Original) - 3);
        Kind := 'q8';
      end
      else if Copy(Original, Length(Original) - 5, 6) = '_scale' then
        begin
          SetLength(Original, Length(Original) - 6);
          Kind := 'scale';
        end;
    end;
    Binding := ModelObject(Bindings.GetValue(Original), Original);
    Tensor := ModelText(Binding.GetValue('tensor'));
    Shape := ModelShape(Binding.GetValue('shape'));
    if Kind = 'q8' then
    begin
      RequireValue(InputValue, Name, 3, Shape);
      Spec := ModelObject(Weights.GetValue(Tensor), Tensor);
      QuantizeWeight('text.' + Original, Folder, Shape, Binding, Spec, QuantizedWeights, ArtifactHashes);
      AddGpuBinding(GpuBindings, 'text.' + Original, Kind);
      Continue;
    end;
    if Kind = 'scale' then
    begin
      RequireCpu((Length(Shape) = 2) and (QuantizedWeights.GetValue('text.' + Original) <> nil),
        'Gemma 3n scale precedes its quantized weights');
      RequireValue(InputValue, Name, 1, [Shape[0]]);
      AddGpuBinding(GpuBindings, 'text.' + Original, Kind);
      Continue;
    end;
    RequireValue(InputValue, Name, 1, Shape);
    if Gpu then AddGpuBinding(GpuBindings, Name, 'original');
    Slice := Binding.GetValue('slice');
    if Copy(Original, Length(Original) - Length(ClipSuffix) + 1, MaxInt) = ClipSuffix then
    begin
      if Weights.GetValue(Name) = nil then
      begin
        RequireCpu(Slice is TJSONNull, 'Unexpected slice for a Gemma 3n correction matrix');
        Spec := ModelObject(Weights.GetValue(Tensor), Tensor);
        SetLength(Values, TensorElements(Shape));
        Stream := TFileStream.Create(BundleMember(Folder, ModelText(Spec.GetValue('file'))),
          fmOpenRead or fmShareDenyWrite);
        try
          Stream.ReadBuffer(Values[0], Length(Values) * SizeOf(Single));
        finally
          Stream.Free;
        end;
        for J := 0 to High(Values) do
          Values[J] := EnsureRange(Values[J], -CoefficientLimit, CoefficientLimit);
        FileName := 'coefficient-' + IntToStr(I - CacheCount - 3) + '.bin';
        Stream := TFileStream.Create(BundleMember(Folder, FileName), fmCreate);
        try
          Stream.WriteBuffer(Values[0], Length(Values) * SizeOf(Single));
          Stream.Position := 0;
          Coefficient := TJSONObject.Create;
          Weights.AddPair(Name, Coefficient);
          Coefficient.AddPair('file', FileName);
          Coefficient.AddPair('dtype', 'float32');
          Coefficient.AddPair('shape', Dimensions(Shape));
          Coefficient.AddPair('sha256', SHA256Stream(Stream));
        finally
          Stream.Free;
        end;
      end;
      Tensor := Name;
    end;
    Spec := TJSONObject.Create;
    Params.AddPair(Name, Spec);
    Spec.AddPair('tensor', Tensor);
    Spec.AddPair('shape', Dimensions(Shape));
    if Slice is TJSONNull then
      Spec.AddPair('slice', TJSONNull.Create)
    else
      Spec.AddPair('slice', JSONInteger(ModelInt(Slice, 'parameter slice')));
  end;
end;

procedure CompileGraph(const Folder, LibraryName: string; const Kit: TWindowsTvmKit; const Model: TOnnxModel;
  const NvrtcLibrary: string = ''; const Architecture: string = '');
var
  Kernel, VM, Output, Extra: string;
  Temporary: TArray<string>;
begin
  Kernel := BundleMember(Folder, 'kernels.obj');
  VM := BundleMember(Folder, 'vm.obj');
  Output := BundleMember(Folder, LibraryName);
  if Architecture = '' then
    CompileCpuOnnx(Model, Kit.CompilerDirectory, WindowsLLVM, Kernel, VM)
  else
    CompileCudaOnnx(Model, Kit.CompilerDirectory, NvrtcLibrary, Architecture, WindowsLLVM, Kernel, VM);
  LinkWindowsTvmLibrary(Kit.Linker, Output, [Kernel, VM], [Kit.FfiLibrary, 'libcmt.lib'],
    [Kit.MsvcLibraries, Kit.UcrtLibraries, Kit.WindowsLibraries]);
  Temporary := [Kernel, VM, ChangeFileExt(Output, '.lib'), ChangeFileExt(Output, '.exp')];
  for Extra in Temporary do
    if FileExists(Extra) then
      RequireCpu(Windows.DeleteFileW(PWideChar(Extra)), 'Could not remove ' + Extra);
end;

procedure CopyRecipe(const Recipe, Stage, Name: string; const Hashes: TJSONObject);
var
  Source, Target: string;
begin
  Source := BundleMember(Recipe, Name);
  Target := BundleMember(Stage, Name);
  if not CopyFileW(PWideChar(Source), PWideChar(Target), True) then RaiseLastOSError;
  RequireCpu(SameText(FileHash(Target), ModelText(Hashes.GetValue(Name))),
    'Gemma 3n graph recipe changed: ' + Name);
end;

function ImageVocabOffset(const Checkpoint: TGGUFCheckpoint): Integer;
var
  Tokens: TArray<UTF8String>;
begin
  Tokens := Checkpoint.TokenPieces('tokenizer.ggml.tokens');
  Result := 0;
  while (Result < Length(Tokens)) and (Tokens[Result] <> '<end_of_image>') do Inc(Result);
  RequireCpu(Result < Length(Tokens), 'The checkpoint has no Gemma 3n image vocabulary');
end;

procedure WriteVisionManifest(const Projector: TGGUFCheckpoint;
  const Folder, ProjectorHash, LibraryName, Architecture: string; const VocabOffset: Integer);
var
  Root, Hashes: TJSONObject;
  Mean, Std: TArray<Single>;
  Name: string;
begin
  Mean := Projector.FloatArray('clip.vision.image_mean');
  Std := Projector.FloatArray('clip.vision.image_std');
  RequireCpu((Length(Mean) = 3) and (Length(Std) = 3) and (Std[0] > 0) and (Std[1] > 0) and (Std[2] > 0),
    'Invalid Gemma 3n projector RGB normalization');
  RequireCpu(VocabOffset >= 0, 'The text model has no Gemma 3n image vocabulary');
  Root := TJSONObject.Create;
  try
    Root.AddPair('schema', 'crosspascal.gemma3n.vision.v1');
    Root.AddPair('architecture', 'gemma3nv');
    if Architecture = '' then
    begin
      Root.AddPair('backend', 'TVM CPU FP32');
      Root.AddPair('cpu_arch', 'haswell');
    end
    else begin
      Root.AddPair('backend', 'TVM CUDA FP32');
      Root.AddPair('cuda_arch', Architecture);
    end;
    Root.AddPair('library', LibraryName);
    Root.AddPair('layout', 'vision-parameters.json');
    Root.AddPair('parameters', 'vision-params.bin');
    Root.AddPair('hard_embeddings', 'vision-hard.npy');
    Root.AddPair('hard_norm', 'vision-hard-norm.npy');
    Root.AddPair('projection', 'vision-projection.npy');
    Root.AddPair('image_size', JSONInteger(Projector.Unsigned('clip.vision.image_size')));
    Root.AddPair('mean', FloatValues(Mean));
    Root.AddPair('std', FloatValues(Std));
    Root.AddPair('rms_norm_eps', JSONFloat(Projector.Float('clip.vision.attention.layer_norm_epsilon')));
    Root.AddPair('soft_tokens', JSONInteger(256));
    Root.AddPair('text_hidden_size', JSONInteger(2048));
    Root.AddPair('vocab_offset', JSONInteger(VocabOffset));
    Root.AddPair('source_sha256', ProjectorHash);
    Hashes := TJSONObject.Create;
    Root.AddPair('artifact_sha256', Hashes);
    Hashes.AddPair(LibraryName, FileHash(BundleMember(Folder, LibraryName)));
    for Name in VisionFiles do Hashes.AddPair(Name, FileHash(BundleMember(Folder, Name)));
    SaveJSON(BundleMember(Folder, 'vision.json'), Root);
  finally
    Root.Free;
  end;
end;

procedure BuildGemma3nVision(const Projector: TGGUFCheckpoint;
  const Stage, ProjectorHash, LibraryName, NvrtcLibrary, Architecture: string; const Kit: TWindowsTvmKit;
  const VocabOffset: Integer; const Manifest: TJSONObject);
var
  Graph: TOnnxModel;
  Modalities: TJSONArray;
  Hashes: TJSONObject;
begin
  Graph := TOnnxModel.Create(BundleMember(Stage, VisionOnnx));
  try
    WriteGemma3nVisionWeights(Projector, Graph, Stage);
    CompileGraph(Stage, LibraryName, Kit, Graph, NvrtcLibrary, Architecture);
  finally
    Graph.Free;
  end;
  WriteVisionManifest(Projector, Stage, ProjectorHash, LibraryName, Architecture, VocabOffset);
  Modalities := ModelArray(Manifest.GetValue('modalities'), 'Gemma 3n modalities');
  Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'Gemma 3n hashes');
  Manifest.AddPair('vision_manifest', 'vision.json');
  Modalities.AddElement(TJSONString.Create('image'));
  Hashes.AddPair('vision.json', FileHash(BundleMember(Stage, 'vision.json')));
end;

procedure DiscardStage(const Folder: string);
var
  Item: TSearchRec;
  Status: Integer;
begin
  Status := FindFirst(IncludeTrailingPathDelimiter(Folder) + '*', faAnyFile, Item);
  if Status = 0 then
    try
      while Status = 0 do
      begin
        if (Item.Attr and faDirectory) = 0 then
          Windows.DeleteFileW(PWideChar(BundleMember(Folder, Item.Name)));
        Status := FindNext(Item);
      end;
    finally
      SysUtils.FindClose(Item);
    end;
  RemoveDir(Folder);
end;

procedure ConvertGemma3nBundle(const Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary, NvrtcLibrary, Architecture: string;
  const Kit: TWindowsTvmKit; const GenerateTextGraphs: Boolean;
  const ContextOverride, ChunkOverride: Integer);
var
  Source, VisionSource: TGGUFCheckpoint;
  Config: TGemma3nConfig;
  Bindings: TArray<TGemma3nBinding>;
  RecipeData, ParameterData: TJSONValue;
  Recipe, RecipeHashes, Manifest, Prefill, Parameters, Weights, Weight, Hashes: TJSONObject;
  Graph: TOnnxModel;
  Modalities: TJSONArray;
  Stage, Output, SourceHash, ProjectorHash, Name, LanguageLibrary, ImageLibrary: string;
  ID: TGUID;
  I, Context, Chunk: Integer;
  Installed, Gpu: Boolean;

  function TextGraph(const Name: string; const TokenCount: Integer): TOnnxModel;
  begin
    if GenerateTextGraphs then
      Exit(BuildGemma3nTextGraph(Config, Bindings, Context, TokenCount, Gpu));
    Result := TOnnxModel.Create(BundleMember(Stage, Name));
  end;

begin
  Gpu := Architecture <> '';
  LanguageLibrary := TextLibrary;
  ImageLibrary := VisionLibrary;
  if Gpu then
  begin
    LanguageLibrary := CudaTextLibrary;
    ImageLibrary := CudaVisionLibrary;
  end;
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu((not DirectoryExists(Output)) and (not FileExists(Output)) and DirectoryExists(ExtractFileDir(Output)),
    'Choose a new bundle directory in an existing parent');
  RecipeData := ReadModelJSON(BundleMember(RecipeDirectory, 'recipe.json'));
  try
    Recipe := ModelObject(RecipeData, 'Gemma 3n graph recipe');
    RequireCpu(ModelText(Recipe.GetValue('schema')) = 'crosspascal.gemma3n.cpu-recipe.v1',
      'Unsupported Gemma 3n recipe');
    RecipeHashes := ModelObject(Recipe.GetValue('graph_sha256'), 'Gemma 3n recipe graphs');
    SourceHash := FileHash(Checkpoint);
    if not GenerateTextGraphs then
      RequireCpu(SameText(SourceHash, ModelText(Recipe.GetValue('source_sha256'))),
        'The Gemma 3n checkpoint does not match the graph recipe');
    Source := TGGUFCheckpoint.Create(Checkpoint, GGMLLibrary);
    VisionSource := nil;
    try
      Bindings := ReadGemma3nBindings(Source, Config);
      Context := ModelInt(Recipe.GetValue('context'), 'compiled context');
      Chunk := ModelInt(Recipe.GetValue('prefill_chunk'), 'prefill chunk');
      if GenerateTextGraphs then
      begin
        if ContextOverride <> 0 then Context := ContextOverride;
        if ChunkOverride <> 0 then Chunk := ChunkOverride;
        if Gpu then Chunk := 2;
      end;
      RequireCpu((Context >= 2) and (Context <= Config.Context) and (Context <= 131072) and (Chunk >= 2) and (Chunk <= Context),
        'Invalid Gemma 3n graph context or prefill chunk');
      if Projector <> '' then
      begin
        ProjectorHash := FileHash(Projector);
        RequireCpu(SameText(ProjectorHash, ModelText(Recipe.GetValue('projector_sha256'))),
          'The Gemma 3n vision projector does not match the graph recipe');
        VisionSource := TGGUFCheckpoint.Create(Projector, GGMLLibrary);
      end;
      RequireCpu(CreateGUID(ID) = 0, 'Could not create a build directory name');
      Stage := Output + '.building-' + GUIDToString(ID);
      RequireCpu(CreateDir(Stage), 'Could not create the Gemma 3n build directory');
      Installed := False;
      Manifest := nil;
      ParameterData := nil;
      try
        if not GenerateTextGraphs then
        begin
          CopyRecipe(RecipeDirectory, Stage, TextOnnx, RecipeHashes);
          CopyRecipe(RecipeDirectory, Stage, PrefillOnnx, RecipeHashes);
        end;
        if VisionSource <> nil then CopyRecipe(RecipeDirectory, Stage, VisionOnnx, RecipeHashes);
        WriteGemma3nTextWeights(Source, SourceHash, SentencePieceLibrary, Stage, Config, Bindings);
        ParameterData := ReadModelJSON(BundleMember(Stage, 'parameters.json'));
        Parameters := ModelObject(ParameterData, 'Gemma 3n parameters');
        Manifest := TJSONObject.Create;
        Manifest.AddPair('schema', 'crosspascal.gemma3n.text.v1');
        Manifest.AddPair('architecture', 'gemma3n');
        if Gpu then
        begin
          Manifest.AddPair('backend', 'TVM CUDA W8A32');
          Manifest.AddPair('cuda_arch', Architecture);
        end
        else
          Manifest.AddPair('backend', 'TVM LLVM CPU FP32');
        Manifest.AddPair('host_embeddings', JSONBoolean(True));
        Manifest.AddPair('library', LanguageLibrary);
        Manifest.AddPair('tokenizer', 'tokenizer.model');
        Manifest.AddPair('source_sha256', SourceHash);
        Manifest.AddPair('context', JSONInteger(Context));
        Manifest.AddPair('context_window_size', JSONInteger(Context));
        Manifest.AddPair('sliding_window', JSONInteger(Config.SlidingWindow));
        Manifest.AddPair('cache_count', JSONInteger(2 * (Config.Layers - Config.SharedLayers)));
        Manifest.AddPair('cache_shape', Dimensions([1, Config.KVHeads, Context, Config.HeadSize]));
        Manifest.AddPair('cpu_arch', 'haswell');
        Modalities := TJSONArray.Create;
        Manifest.AddPair('modalities', Modalities);
        Modalities.AddElement(TJSONString.Create('text'));
        Hashes := TJSONObject.Create;
        Manifest.AddPair('artifact_sha256', Hashes);
        Graph := TextGraph(TextOnnx, 1);
        try
          BindTextGraph(Graph, Config, Context, 1, Stage, Parameters, Manifest, Gpu, Hashes);
          CompileGraph(Stage, LanguageLibrary, Kit, Graph, NvrtcLibrary, Architecture);
        finally
          Graph.Free;
        end;
        if not Gpu then
        begin
          Prefill := TJSONObject.Create;
          Manifest.AddPair('prefill', Prefill);
          Prefill.AddPair('chunk', JSONInteger(Chunk));
          if not GenerateTextGraphs then Prefill.AddPair('onnx', PrefillOnnx);
          Prefill.AddPair('library', PrefillLibrary);
          Graph := TextGraph(PrefillOnnx, Chunk);
          try
            BindTextGraph(Graph, Config, Context, Chunk, Stage, Parameters, Prefill);
            CompileGraph(Stage, PrefillLibrary, Kit, Graph);
          finally
            Graph.Free;
          end;
          Hashes.AddPair(PrefillLibrary, FileHash(BundleMember(Stage, PrefillLibrary)));
        end;
        SaveJSON(BundleMember(Stage, 'parameters.json'), Parameters);
        if not GenerateTextGraphs then
        begin
          Hashes.AddPair(TextOnnx, FileHash(BundleMember(Stage, TextOnnx)));
          Hashes.AddPair(PrefillOnnx, FileHash(BundleMember(Stage, PrefillOnnx)));
        end;
        Hashes.AddPair(LanguageLibrary, FileHash(BundleMember(Stage, LanguageLibrary)));
        Hashes.AddPair('parameters.json', FileHash(BundleMember(Stage, 'parameters.json')));
        Hashes.AddPair('tokenizer.model', FileHash(BundleMember(Stage, 'tokenizer.model')));
        Weights := ModelObject(Parameters.GetValue('weights'), 'Gemma 3n weights');
        for I := 0 to Weights.Count - 1 do
        begin
          Weight := ModelObject(JSONObjectValue(Weights, I), 'Gemma 3n weight');
          Name := ModelText(Weight.GetValue('file'));
          Hashes.AddPair(Name, ModelText(Weight.GetValue('sha256')));
        end;
        if VisionSource <> nil then
          BuildGemma3nVision(VisionSource, Stage, ProjectorHash, ImageLibrary, NvrtcLibrary,
            Architecture, Kit, ImageVocabOffset(Source), Manifest);
        SaveJSON(BundleMember(Stage, 'model.json'), Manifest);
        RequireCpu(RenameFile(Stage, Output), 'Could not install the new Gemma 3n bundle');
        Installed := True;
      finally
        ParameterData.Free;
        Manifest.Free;
        if not Installed then DiscardStage(Stage);
      end;
    finally
      VisionSource.Free;
      Source.Free;
    end;
  finally
    RecipeData.Free;
  end;
end;

procedure ConvertGemma3nCpuBundle(const Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary: string;
  const Kit: TWindowsTvmKit; const GenerateTextGraphs: Boolean;
  const ContextOverride, ChunkOverride: Integer);
begin
  ConvertGemma3nBundle(Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary,
    SentencePieceLibrary, '', '', Kit, GenerateTextGraphs, ContextOverride, ChunkOverride);
end;

procedure ConvertGemma3nCudaBundle(const Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary, NvrtcLibrary, Architecture: string;
  const Kit: TWindowsTvmKit; const ContextOverride: Integer);
begin
  RequireCpu((NvrtcLibrary <> '') and (Architecture <> ''), 'A CUDA architecture and NVRTC are required');
  ConvertGemma3nBundle(Checkpoint, Projector, RecipeDirectory, Destination, GGMLLibrary,
    SentencePieceLibrary, NvrtcLibrary, Architecture, Kit, True, ContextOverride, 0);
end;

procedure AddGemma3nVision(const TextBundle, Projector, RecipeDirectory, Destination, GGMLLibrary, SentencePieceLibrary, NvrtcLibrary: string;
  const Kit: TWindowsTvmKit);
var
  TextData, RecipeData, ParameterData: TJSONValue;
  Manifest, Recipe, Hashes, Weights, Spec: TJSONObject;
  Modalities: TJSONArray;
  Source: TGGUFCheckpoint;
  Tokenizer: TSentencePiece;
  Output, Stage, Name, Target, Backend, Architecture, LibraryName, ProjectorHash: string;
  ID: TGUID;
  I: Integer;

  procedure CopyArtifact(const Member, ExpectedHash: string);
  var
    Path, ActualHash: string;
  begin
    Path := BundleMember(Stage, Member);
    if not FileExists(Path) then
      if not CopyFileW(PWideChar(BundleMember(TextBundle, Member)), PWideChar(Path), True) then
        RaiseLastOSError;
    ActualHash := FileHash(Path);
    RequireCpu((ExpectedHash = '') or SameText(ActualHash, ExpectedHash),
      'A Gemma 3n text artifact changed: ' + Member);
    if Hashes.GetValue(Member) = nil then Hashes.AddPair(Member, ActualHash);
  end;

begin
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu(DirectoryExists(ExtractFileDir(Output)) and not DirectoryExists(Output) and
    not FileExists(Output), 'Choose a new bundle directory in an existing parent');
  TextData := ReadModelJSON(BundleMember(TextBundle, 'model.json'));
  RecipeData := nil;
  ParameterData := nil;
  Source := nil;
  Tokenizer := nil;
  Stage := '';
  try
    Manifest := ModelObject(TextData, 'Gemma 3n text manifest');
    RequireCpu(
      (ModelText(Manifest.GetValue('schema')) = 'crosspascal.gemma3n.text.v1') and JSONIsTrue(Manifest.GetValue('host_embeddings')) and (Manifest.GetValue('vision_manifest') = nil),
      'Expected a native text-only Gemma 3n bundle with host embeddings'
    );
    Modalities := ModelArray(Manifest.GetValue('modalities'), 'Gemma 3n modalities');
    RequireCpu((Modalities.Count = 1) and (ModelText(Modalities.Items[0]) = 'text'),
      'The Gemma 3n bundle is not text-only');
    Backend := ModelText(Manifest.GetValue('backend'));
    Architecture := '';
    LibraryName := VisionLibrary;
    if Backend = 'TVM CUDA W8A32' then
    begin
      Architecture := ModelText(Manifest.GetValue('cuda_arch'));
      RequireCpu((Architecture <> '') and (NvrtcLibrary <> ''), 'The CUDA text bundle needs NVIDIA NVRTC');
      LibraryName := CudaVisionLibrary;
    end
    else
      RequireCpu(Backend = 'TVM LLVM CPU FP32', 'Unsupported Gemma 3n text backend');
    RecipeData := ReadModelJSON(BundleMember(RecipeDirectory, 'recipe.json'));
    Recipe := ModelObject(RecipeData, 'Gemma 3n graph recipe');
    RequireCpu(ModelText(Recipe.GetValue('schema')) = 'crosspascal.gemma3n.cpu-recipe.v1',
      'Unsupported Gemma 3n recipe');
    ProjectorHash := FileHash(Projector);
    RequireCpu(SameText(ProjectorHash, ModelText(Recipe.GetValue('projector_sha256'))),
      'The projector does not match the Gemma 3n graph recipe');
    Source := TGGUFCheckpoint.Create(Projector, GGMLLibrary);
    RequireCpu(CreateGUID(ID) = 0, 'Could not name the Gemma 3n build directory');
    Target := Output + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(Target), 'Could not create the Gemma 3n build directory');
    Stage := Target;
    Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'Gemma 3n artifact hashes');
    RequireCpu((Hashes.GetValue('tokenizer.model') <> nil) and (Hashes.GetValue('parameters.json') <> nil),
      'The Gemma 3n text bundle has no verified tokenizer or parameters');
    for I := 0 to Hashes.Count - 1 do
    begin
      Name := JSONObjectName(Hashes, I);
      CopyArtifact(Name, ModelText(JSONObjectValue(Hashes, I)));
    end;
    ParameterData := ReadModelJSON(BundleMember(Stage, 'parameters.json'));
    Weights := ModelObject(ModelObject(ParameterData, 'Gemma 3n parameters').GetValue('weights'),
      'Gemma 3n weights');
    for I := 0 to Weights.Count - 1 do
    begin
      Spec := ModelObject(JSONObjectValue(Weights, I), 'Gemma 3n parameter');
      Name := ModelText(Spec.GetValue('file'));
      if Hashes.GetValue(Name) = nil then CopyArtifact(Name, ModelText(Spec.GetValue('sha256')));
    end;
    if Architecture <> '' then
    begin
      Weights := ModelObject(ModelObject(Manifest.GetValue('quantization'), 'Gemma 3n quantization').GetValue('weights'),
        'Gemma 3n quantized weights');
      for I := 0 to Weights.Count - 1 do
      begin
        Spec := ModelObject(JSONObjectValue(Weights, I), 'Gemma 3n quantized parameter');
        Name := ModelText(Spec.GetValue('q8'));
        if Hashes.GetValue(Name) = nil then CopyArtifact(Name, '');
        Name := ModelText(Spec.GetValue('scale'));
        if Hashes.GetValue(Name) = nil then CopyArtifact(Name, '');
      end;
    end;
    Tokenizer := TSentencePiece.Create(SentencePieceLibrary, BundleMember(Stage, 'tokenizer.model'));
    RequireCpu((Tokenizer.ID('<start_of_image>') >= 0) and (Tokenizer.ID('<image_soft_token>') >= 0),
      'The text tokenizer has no image tokens');
    CopyRecipe(RecipeDirectory, Stage, VisionOnnx,
      ModelObject(Recipe.GetValue('graph_sha256'), 'graph hashes'));
    BuildGemma3nVision(Source, Stage, ProjectorHash, LibraryName, NvrtcLibrary, Architecture, Kit,
      Tokenizer.ID('<end_of_image>'), Manifest);
    SaveJSON(BundleMember(Stage, 'model.json'), Manifest);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(Output)), 'Could not install the Gemma 3n image bundle');
    Stage := '';
  finally
    if Stage <> '' then DiscardStage(Stage);
    Tokenizer.Free;
    Source.Free;
    ParameterData.Free;
    RecipeData.Free;
    TextData.Free;
  end;
end;

end.
