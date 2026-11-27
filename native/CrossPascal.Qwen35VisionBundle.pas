{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Qwen35VisionBundle                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Qwen35VisionBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.TVMCompiler;

{ Writes a standalone Qwen3.5 vision bundle into Destination, which must be an
  existing empty directory: vision-parameters.bin (unchanged source BF16 bytes of
  every graph weight, in graph input order), vision.dll (TVM library of
  BuildQwen35VisionGraph) and vision.json (manifest, written last). SourceShard is
  the single safetensors shard that holds every model.visual.* tensor. CPU when
  NvrtcLibrary and Architecture are both empty, CUDA when both are given. On failure
  the outputs created by this call are removed and no manifest remains. }
procedure WriteQwen35VisionBundle(const Root: TJSONObject; const SourceShard, Destination: string;
  const ImageHeight, ImageWidth: Integer; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string);

implementation

uses
  {$IFDEF FPC}SysUtils, Classes, Generics.Collections
  {$ELSE}System.SysUtils, System.Classes, System.Generics.Collections{$ENDIF},
  CrossPascal.Onnx, CrossPascal.TVMOnnx, CrossPascal.Qwen35VisionGraph, CrossPascal.SafeTensors,
  CrossPascal.SHA256, TVM.CpuGraph, CrossPascal.NativeIO;

const
  ManifestSchema = 'crosspascal-qwen35-vision-v1';
  ManifestFile = 'vision.json';
  LibraryFile = 'vision.dll';
  ParametersFile = 'vision-parameters.bin';
  PatchesInput = 'patches';
  EmbeddingsOutput = 'image_embeddings';
  VisualPrefix = 'model.visual.';
  { TQwen35WeightFiles.Weight: 'bf16' -> Format 30, BlockElements 1, BlockBytes 2. }
  WeightEncoding = 'bf16';
  SourceDType = 'BF16';
  BF16Bytes = 2;
  { TVM.CpuGraph.ModelShape accepts ranks 1..8. }
  MaxWeightRank = 8;
  MaxArchitectureLength = 32;

type
  TVisionWeight = record
    Name: string;
    Shape: TArray<Int64>;
    Elements: Int64;
    Bytes: Int64;
    Offset: Int64;
  end;

procedure RequireFreshDestination(const Folder: string);
var
  {$IFDEF FPC}
  Search: TUnicodeSearchRec;
  {$ELSE}
  Search: TSearchRec;
  {$ENDIF}
begin
  RequireCpu(DirectoryExists(Folder), 'Qwen vision destination does not exist');
  if FindFirst(IncludeTrailingPathDelimiter(Folder) + '*', faAnyFile, Search) = 0 then
    try
      repeat
        RequireCpu((Search.Name = '.') or (Search.Name = '..'),
          'Qwen vision destination must be a new empty directory');
      until FindNext(Search) <> 0;
    finally
      FindClose(Search);
    end;
end;

function ValidArchitecture(const Value: string): Boolean;
var
  I: Integer;
  C: Char;
begin
  Result := (Value <> '') and (Length(Value) <= MaxArchitectureLength);
  if not Result then Exit;
  for I := 1 to Length(Value) do
  begin
    C := Value[I];
    if not (((C >= 'a') and (C <= 'z')) or ((C >= '0') and (C <= '9')) or (C = '_')) then
      Exit(False);
  end;
end;

function ValidWeightName(const Name: string): Boolean;
var
  I: Integer;
begin
  Result := (Length(Name) > Length(VisualPrefix)) and
    (Copy(Name, 1, Length(VisualPrefix)) = VisualPrefix);
  if not Result then Exit;
  for I := 1 to Length(Name) do
    if Name[I] < ' ' then Exit(False);
end;

function SameShape(const Left, Right: array of Int64): Boolean;
var
  I: Integer;
begin
  Result := Length(Left) = Length(Right);
  if not Result then Exit;
  for I := 0 to High(Left) do
    if Left[I] <> Right[I] then Exit(False);
end;

function GraphShape(const Value: TOnnxValue): TArray<Int64>;
var
  I: Integer;
begin
  RequireCpu(Value.HasShape and (Length(Value.Shape) > 0) and (Length(Value.Shape) <= MaxWeightRank),
    'Qwen vision graph value has no static shape: ' + Value.Name);
  SetLength(Result, Length(Value.Shape));
  for I := 0 to High(Result) do
  begin
    RequireCpu((Value.Shape[I].Symbol = '') and (Value.Shape[I].Value > 0),
      'Qwen vision graph value has a dynamic or empty dimension: ' + Value.Name);
    Result[I] := Value.Shape[I].Value;
  end;
  { Positive dimensions, product at most High(Int64) div 8. }
  TensorElements(Result);
end;

procedure HashFile(const FileName: string; out Size: Int64; out Digest: string);
var
  Stream: TStream;
  I: Integer;
begin
  Stream := TNativeReadStream.Create(FileName);
  try
    Size := Stream.Size;
    Digest := LowerCase(string(SHA256Stream(Stream)));
  finally
    Stream.Free;
  end;
  RequireCpu(Length(Digest) = 64, 'Unexpected SHA-256 digest length');
  for I := 1 to Length(Digest) do
    RequireCpu(((Digest[I] >= '0') and (Digest[I] <= '9')) or
      ((Digest[I] >= 'a') and (Digest[I] <= 'f')), 'Unexpected SHA-256 digest text');
end;

function ShapeJSON(const Shape: array of Int64): TJSONArray;
var
  I: Integer;
begin
  Result := TJSONArray.Create;
  try
    for I := 0 to High(Shape) do Result.AddElement(JSONInteger(Shape[I]));
  except
    Result.Free;
    raise;
  end;
end;

procedure WriteManifestFile(const FileName: string; const Manifest: TJSONObject);
var
  Text: UTF8String;
  Stream: TStream;
begin
  Text := UTF8Encode(Manifest.ToJSON);
  RequireCpu(Length(Text) > 0, 'Empty Qwen vision manifest');
  Stream := TNativeWriteStream.Create(FileName, False);
  try
    Stream.WriteBuffer(Text[1], Length(Text));
    RequireCpu(Stream.Size = Length(Text), 'Incomplete Qwen vision manifest');
  finally
    Stream.Free;
  end;
end;

procedure RemoveIfCreated(const Created: Boolean; const FileName: string);
begin
  if Created and FileExists(FileName) then DeleteFile(FileName);
end;

procedure WriteQwen35VisionBundle(const Root: TJSONObject; const SourceShard, Destination: string;
  const ImageHeight, ImageWidth: Integer; const Kit: TWindowsTvmKit;
  const NvrtcLibrary, Architecture: string);
var
  Folder, SourcePath, ParametersPath, LibraryPath, ManifestPath, Name: string;
  SourceDigest, ParametersDigest, LibraryDigest: string;
  Model: TOnnxModel;
  Source: TSafeTensorFile;
  Info: TSafeTensorInfo;
  Storage: TSafeTensorStorage;
  Seen: TDictionary<string, Boolean>;
  Weights: TArray<TVisionWeight>;
  Value: TOnnxValue;
  Shape: TArray<Int64>;
  Output: TStream;
  Vision, Text, Manifest, Entry, Descriptor, WeightMap: TJSONObject;
  Inputs, Files: TJSONArray;
  Cuda, CreatedParameters, CreatedLibrary, CreatedManifest: Boolean;
  I, VisualCount: Integer;
  Channels, Temporal, Patch, Merge, Hidden, GridH, GridW, Tokens: Int64;
  Total, Start, SourceSize, ParametersSize, LibrarySize: Int64;
begin
  RequireCpu(Root <> nil, 'Qwen config is missing');
  Cuda := NvrtcLibrary <> '';
  { Qwen35Bundle selects the device by NvrtcLibrary, CompileWindowsTvmOnnxLibrary by
    Architecture; both must agree. Architecture is embedded in a JSON target string. }
  RequireCpu(Cuda = (Architecture <> ''),
    'Qwen vision CUDA build needs both an NVRTC library and an architecture');
  if Cuda then
    RequireCpu(ValidArchitecture(Architecture), 'Invalid CUDA architecture name');
  Folder := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireFreshDestination(Folder);
  SourcePath := ExpandFileName(SourceShard);
  RequireCpu(FileExists(SourcePath), 'Qwen vision source shard does not exist');
  ParametersPath := BundleMember(Folder, ParametersFile);
  LibraryPath := BundleMember(Folder, LibraryFile);
  ManifestPath := BundleMember(Folder, ManifestFile);
  RequireCpu(not FileExists(ParametersPath) and not FileExists(LibraryPath) and
    not FileExists(ManifestPath), 'Qwen vision output already exists');
  Model := nil;
  Source := nil;
  Seen := nil;
  Manifest := nil;
  CreatedParameters := False;
  CreatedLibrary := False;
  CreatedManifest := False;
  try
    try
      { Validates the config and image size; graph order is the bundle order. }
      Model := BuildQwen35VisionGraph(Root, ImageHeight, ImageWidth);
      Vision := ModelObject(Root.GetValue('vision_config'), 'vision_config');
      Text := ModelObject(Root.GetValue('text_config'), 'text_config');
      Channels := ModelInt(Vision.GetValue('in_channels'), 'in_channels');
      Temporal := ModelInt(Vision.GetValue('temporal_patch_size'), 'temporal_patch_size');
      Patch := ModelInt(Vision.GetValue('patch_size'), 'patch_size');
      Merge := ModelInt(Vision.GetValue('spatial_merge_size'), 'spatial_merge_size');
      Hidden := ModelInt(Text.GetValue('hidden_size'), 'hidden_size');
      RequireCpu((Patch > 0) and (Merge > 0) and (Temporal > 0) and (Channels > 0) and (Hidden > 0),
        'Invalid Qwen vision configuration');
      GridH := ImageHeight div Patch;
      GridW := ImageWidth div Patch;
      Tokens := TensorElements([GridH, GridW]);

      { Graph contract: patches first, one image_embeddings output. }
      RequireCpu(Length(Model.Inputs) >= 2, 'Qwen vision graph has no weight inputs');
      Value := Model.Inputs[0];
      RequireCpu((Value.Name = PatchesInput) and (Value.DataType = OnnxFloat),
        'Qwen vision graph must begin with the FP32 patches input');
      Shape := GraphShape(Value);
      RequireCpu(SameShape(Shape, [Tokens, TensorElements([Channels, Temporal, Patch, Patch])]),
        'Qwen vision patches shape differs from the configuration');
      RequireCpu(Length(Model.Outputs) = 1, 'Qwen vision graph must have one output');
      Value := Model.Outputs[0];
      RequireCpu((Value.Name = EmbeddingsOutput) and (Value.DataType = OnnxFloat),
        'Qwen vision graph output must be FP32 image_embeddings');
      Shape := GraphShape(Value);
      RequireCpu(SameShape(Shape, [Tokens div (Merge * Merge), Hidden]),
        'Qwen vision embeddings shape differs from the configuration');

      { Validate every weight before writing anything. }
      Source := TSafeTensorFile.Create(SourcePath);
      Seen := TDictionary<string, Boolean>.Create;
      Seen.Add(PatchesInput, True);
      SetLength(Weights, Length(Model.Inputs) - 1);
      Total := 0;
      for I := 1 to High(Model.Inputs) do
      begin
        Value := Model.Inputs[I];
        Name := Value.Name;
        RequireCpu(ValidWeightName(Name), 'Unexpected Qwen vision graph input: ' + Name);
        RequireCpu(not Seen.ContainsKey(Name), 'Duplicate Qwen vision graph input: ' + Name);
        Seen.Add(Name, True);
        RequireCpu(Value.DataType = OnnxFloat, 'Qwen vision weight input is not FP32: ' + Name);
        Weights[I - 1].Name := Name;
        Weights[I - 1].Shape := GraphShape(Value);
        Weights[I - 1].Elements := TensorElements(Weights[I - 1].Shape);
        { TensorElements bounds the product by High(Int64) div 8. }
        Weights[I - 1].Bytes := Weights[I - 1].Elements * BF16Bytes;
        Info := Source.TensorInfo(Name);
        RequireCpu(Info.DType = SourceDType, 'Qwen vision source tensor is not BF16: ' + Name);
        RequireCpu(SameShape(Info.Shape, Weights[I - 1].Shape) and
          (Info.ElementCount = Weights[I - 1].Elements),
          'Qwen vision source shape differs from the graph: ' + Name);
        Storage := Source.TensorStorage(Name);
        RequireCpu(Storage.ByteCount = Weights[I - 1].Bytes,
          'Qwen vision source byte count differs: ' + Name);
        RequireCpu(Total <= High(Int64) - Weights[I - 1].Bytes, 'Qwen vision parameters are too large');
        Weights[I - 1].Offset := Total;
        Inc(Total, Weights[I - 1].Bytes);
      end;
      { The graph and the shard must describe the same visual tensor set. }
      VisualCount := 0;
      for Name in Source.TensorNames do
        if Copy(Name, 1, Length(VisualPrefix)) = VisualPrefix then
        begin
          RequireCpu(Seen.ContainsKey(Name), 'Source visual tensor is not used by the graph: ' + Name);
          Inc(VisualCount);
        end;
      RequireCpu(VisualCount = Length(Weights), 'Qwen vision graph and source tensor counts differ');
      { TSafeTensorFile holds the shard with fmShareDenyWrite, so this digest covers the
        exact bytes copied below. }
      HashFile(SourcePath, SourceSize, SourceDigest);
      RequireCpu(SourceSize = Source.TensorStorage(Weights[0].Name).FileSize,
        'Qwen vision source shard size changed');
      RequireCpu(not FileExists(ParametersPath), 'Qwen vision parameters already exist');
      CreatedParameters := True;
      Output := TNativeWriteStream.Create(ParametersPath, False);
      try
        for I := 0 to High(Weights) do
        begin
          Start := Output.Position;
          RequireCpu(Start = Weights[I].Offset, 'Qwen vision parameter offset mismatch');
          Source.CopyRawElements(Weights[I].Name, 0, Weights[I].Elements, Output);
          RequireCpu(Output.Position - Start = Weights[I].Bytes,
            'Incomplete Qwen vision parameter copy: ' + Weights[I].Name);
          Weights[I].Offset := Start;
        end;
        RequireCpu((Output.Position = Total) and (Output.Size = Total),
          'Qwen vision parameter file size mismatch');
      finally
        Output.Free;
      end;
      FreeAndNil(Source);
      RequireCpu(not FileExists(LibraryPath), 'Qwen vision library already exists');
      CreatedLibrary := True;
      CompileWindowsTvmOnnxLibrary(Model, Kit, LibraryPath, NvrtcLibrary, Architecture);
      RequireCpu(FileExists(LibraryPath), 'Qwen vision library was not produced');
      HashFile(ParametersPath, ParametersSize, ParametersDigest);
      RequireCpu(ParametersSize = Total, 'Qwen vision parameter file size changed');
      HashFile(LibraryPath, LibrarySize, LibraryDigest);
      RequireCpu(LibrarySize > 0, 'Qwen vision library is empty');
      Manifest := TJSONObject.Create;
      Manifest.AddPair('schema', ManifestSchema);
      if Cuda then
      begin
        Manifest.AddPair('device', 'cuda');
        Manifest.AddPair('cuda_arch', Architecture);
      end
      else
        Manifest.AddPair('device', 'cpu');
      Manifest.AddPair('image_height', JSONInteger(ImageHeight));
      Manifest.AddPair('image_width', JSONInteger(ImageWidth));
      Manifest.AddPair('patch_size', JSONInteger(Patch));
      Manifest.AddPair('temporal_patch_size', JSONInteger(Temporal));
      Manifest.AddPair('spatial_merge_size', JSONInteger(Merge));
      Manifest.AddPair('grid_height', JSONInteger(GridH));
      Manifest.AddPair('grid_width', JSONInteger(GridW));
      Manifest.AddPair('hidden_size', JSONInteger(Hidden));
      Manifest.AddPair('patches_input', PatchesInput);
      Manifest.AddPair('embeddings_output', EmbeddingsOutput);
      Manifest.AddPair('library', LibraryFile);
      Manifest.AddPair('source_sha256', SourceDigest);
      Manifest.AddPair('source_file_size', JSONInteger(SourceSize));
      Manifest.AddPair('weights_file', ParametersFile);
      Inputs := TJSONArray.Create;
      Manifest.AddPair('inputs', Inputs);
      for I := 0 to High(Model.Inputs) do
      begin
        Value := Model.Inputs[I];
        Entry := TJSONObject.Create;
        Inputs.AddElement(Entry);
        Entry.AddPair('name', Value.Name);
        Entry.AddPair('dtype', JSONInteger(Value.DataType));
        Entry.AddPair('shape', ShapeJSON(GraphShape(Value)));
      end;
      WeightMap := TJSONObject.Create;
      Manifest.AddPair('weights', WeightMap);
      for I := 0 to High(Weights) do
      begin
        Descriptor := TJSONObject.Create;
        WeightMap.AddPair(Weights[I].Name, Descriptor);
        Descriptor.AddPair('file', ParametersFile);
        Descriptor.AddPair('encoding', WeightEncoding);
        Descriptor.AddPair('offset', JSONInteger(Weights[I].Offset));
        Descriptor.AddPair('bytes', JSONInteger(Weights[I].Bytes));
        Descriptor.AddPair('elements', JSONInteger(Weights[I].Elements));
        Descriptor.AddPair('shape', ShapeJSON(Weights[I].Shape));
      end;
      Files := TJSONArray.Create;
      Manifest.AddPair('files', Files);
      Entry := TJSONObject.Create;
      Files.AddElement(Entry);
      Entry.AddPair('file', LibraryFile);
      Entry.AddPair('bytes', JSONInteger(LibrarySize));
      Entry.AddPair('sha256', LibraryDigest);
      Entry := TJSONObject.Create;
      Files.AddElement(Entry);
      Entry.AddPair('file', ParametersFile);
      Entry.AddPair('bytes', JSONInteger(ParametersSize));
      Entry.AddPair('sha256', ParametersDigest);
      RequireCpu(not FileExists(ManifestPath), 'Qwen vision manifest already exists');
      CreatedManifest := True;
      WriteManifestFile(ManifestPath, Manifest);
    except
      FreeAndNil(Source);
      RemoveIfCreated(CreatedManifest, ManifestPath);
      RemoveIfCreated(CreatedLibrary, LibraryPath);
      RemoveIfCreated(CreatedParameters, ParametersPath);
      raise;
    end;
  finally
    Manifest.Free;
    Seen.Free;
    Source.Free;
    Model.Free;
  end;
end;

end.
