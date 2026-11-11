{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma2Bundle                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma2Bundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.TVMCompiler;

procedure ConvertGemma2Bundle(const Checkpoint, Destination: string; const Context, Chunk: Integer;
  const Kit: TWindowsTvmKit);

implementation

uses
  {$IFDEF FPC}Windows, SysUtils, Classes{$ELSE}Winapi.Windows, System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.Gemma2Graph, CrossPascal.Json, CrossPascal.Onnx, CrossPascal.SafeTensors,
  CrossPascal.SHA256, CrossPascal.TokenizerSource, CrossPascal.TVMOnnx, TVM.CpuGraph;

const
  DecodeLibrary = 'gemma2-cpu.dll';
  PrefillLibrary = 'gemma2-prefill-cpu.dll';
  ParameterFile = 'weights.bin';
  TokenizerMembers: array[0..6] of string = ('tokenizer-tokens.bin', 'tokenizer-tokens.index',
    'tokenizer-merges.bin', 'tokenizer-chars.bin', 'tokenizer-bytes.bin', 'tokenizer-specials.txt',
    'tokenizer.index');

function Dimensions(const Shape: array of Int64): TJSONArray;
var
  Size: Int64;
begin
  Result := TJSONArray.Create;
  for Size in Shape do Result.AddElement(JSONInteger(Size));
end;

function HashFile(const FileName: string): string;
var
  Stream: TFileStream;
begin
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    Result := SHA256Stream(Stream);
  finally
    Stream.Free;
  end;
end;

procedure WriteManifest(const Folder: string; const Manifest: TJSONObject);
var
  Stream: TFileStream;
  Text: UTF8String;
begin
  Text := UTF8Encode(Manifest.ToJSON + #10);
  Stream := TFileStream.Create(BundleMember(Folder, 'model.json'), fmCreate);
  try
    Stream.WriteBuffer(Text[1], Length(Text));
  finally
    Stream.Free;
  end;
end;

procedure CompileGraph(const Folder, LibraryName: string; const Kit: TWindowsTvmKit;
  const Graph: TOnnxModel; const Hashes: TJSONObject);
var
  LibraryPath: string;
begin
  LibraryPath := BundleMember(Folder, LibraryName);
  CompileWindowsTvmOnnxLibrary(Graph, Kit, LibraryPath);
  Hashes.AddPair(LibraryName, HashFile(LibraryPath));
end;

procedure WriteWeights(const Folder: string; const Checkpoint: TSafeTensorCheckpoint;
  const Decode, Prefill: TOnnxModel; const RuntimeInputs: Integer; const Layout: TJSONArray);
var
  Stream: TFileStream;
  Weight, Other: TOnnxValue;
  Shape: TArray<Int64>;
  Entry: TJSONObject;
  Offset, Expected: Int64;
  I, J: Integer;
begin
  RequireCpu(Length(Decode.Inputs) = Length(Prefill.Inputs), 'Gemma 2 graph input count changed');
  Stream := TFileStream.Create(BundleMember(Folder, ParameterFile), fmCreate);
  try
    for I := RuntimeInputs to High(Decode.Inputs) do
    begin
      Weight := Decode.Inputs[I];
      Other := Prefill.Inputs[I];
      RequireCpu((Weight.Name = Other.Name) and (Length(Weight.Shape) = Length(Other.Shape)),
        'Gemma 2 weight binding order changed');
      SetLength(Shape, Length(Weight.Shape));
      for J := 0 to High(Shape) do
      begin
        Shape[J] := Weight.Shape[J].Value;
        RequireCpu((Shape[J] > 0) and (Shape[J] = Other.Shape[J].Value),
          'Gemma 2 weight shape changed: ' + Weight.Name);
      end;
      Offset := Stream.Position;
      Expected := TensorElements(Shape) * SizeOf(Single);
      Checkpoint.WriteFloat32(Weight.Name, Stream);
      RequireCpu(Stream.Position - Offset = Expected, 'Incorrect Gemma 2 parameter size: ' + Weight.Name);
      Entry := TJSONObject.Create;
      Layout.AddElement(Entry);
      Entry.AddPair('name', Weight.Name);
      Entry.AddPair('offset', JSONInteger(Offset));
      Entry.AddPair('shape', Dimensions(Shape));
    end;
  finally
    Stream.Free;
  end;
end;

procedure RemoveStage(const Folder: string);
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
          DeleteFileW(PWideChar(BundleMember(Folder, Item.Name)));
        Status := FindNext(Item);
      end;
    finally
      SysUtils.FindClose(Item);
    end;
  RemoveDir(Folder);
end;

procedure ConvertGemma2Bundle(const Checkpoint, Destination: string; const Context, Chunk: Integer;
  const Kit: TWindowsTvmKit);
var
  Data: TJSONValue;
  Source, Manifest, Hashes: TJSONObject;
  SourceEOS, EOS, Layout: TJSONArray;
  Weights: TSafeTensorCheckpoint;
  Decode, Prefill: TOnnxModel;
  Config: TGemma2Config;
  Stage, Output, Name: string;
  ID: TGUID;
  Published: Boolean;
  I: Integer;
  TokenID: Int64;
begin
  Output := ExcludeTrailingPathDelimiter(ExpandFileName(Destination));
  RequireCpu(DirectoryExists(ExtractFileDir(Output)) and not DirectoryExists(Output) and
    not FileExists(Output), 'Choose a new bundle directory in an existing parent');
  Data := ReadModelJSON(BundleMember(Checkpoint, 'config.json'));
  Weights := nil;
  Manifest := nil;
  Decode := nil;
  Prefill := nil;
  Stage := '';
  Published := False;
  try
    Source := ModelObject(Data, 'Gemma 2 checkpoint configuration');
    Config := ReadGemma2Config(Source);
    RequireCpu((Context >= 2) and (Context <= Config.MaxContext) and (Chunk >= 2) and
      (Chunk <= Context), 'Invalid Gemma 2 context or prefill chunk');
    Weights := TSafeTensorCheckpoint.Create(Checkpoint);
    Decode := BuildGemma2Graph(Config, Weights, Context, 1);
    Prefill := BuildGemma2Graph(Config, Weights, Context, Chunk);
    RequireCpu(CreateGUID(ID) = 0, 'Could not name the Gemma 2 build directory');
    Name := Output + '.building-' + GUIDToString(ID);
    RequireCpu(CreateDir(Name), 'Could not create the Gemma 2 build directory');
    Stage := Name;
    Manifest := TJSONObject.Create;
    Manifest.AddPair('schema', 'crosspascal.gemma2.text.v1');
    Manifest.AddPair('model_type', 'gemma2');
    Manifest.AddPair('target', 'cpu');
    Manifest.AddPair('library', DecodeLibrary);
    Manifest.AddPair('prefill_library', PrefillLibrary);
    Manifest.AddPair('context_window_size', JSONInteger(Context));
    Manifest.AddPair('prefill_chunk_size', JSONInteger(Chunk));
    Manifest.AddPair('cache_count', JSONInteger(2 * Config.Layers));
    Manifest.AddPair('cache_shape', Dimensions([1, Config.KVHeads, Context, Config.HeadSize]));
    Manifest.AddPair('vocab_size', JSONInteger(Config.Vocab));
    TokenID := ModelInt(Source.GetValue('bos_token_id'), 'BOS token');
    RequireCpu((TokenID >= 0) and (TokenID < Config.Vocab), 'Gemma 2 BOS token is out of range');
    Manifest.AddPair('bos_token_id', JSONInteger(TokenID));
    SourceEOS := ModelArray(Source.GetValue('eos_token_id'), 'EOS tokens');
    EOS := TJSONArray.Create;
    Manifest.AddPair('eos_token_id', EOS);
    RequireCpu(SourceEOS.Count > 0, 'Gemma 2 checkpoint has no EOS token');
    for I := 0 to SourceEOS.Count - 1 do
    begin
      TokenID := ModelInt(SourceEOS.Items[I], 'EOS token');
      RequireCpu((TokenID >= 0) and (TokenID < Config.Vocab), 'Gemma 2 EOS token is out of range');
      EOS.AddElement(JSONInteger(TokenID));
    end;
    Layout := TJSONArray.Create;
    Manifest.AddPair('weights', Layout);
    Hashes := TJSONObject.Create;
    Manifest.AddPair('artifact_sha256', Hashes);
    ExportGemmaTokenizer(BundleMember(Checkpoint, 'tokenizer.json'), Stage, Config.Vocab);
    for Name in TokenizerMembers do Hashes.AddPair(Name, HashFile(BundleMember(Stage, Name)));
    WriteWeights(Stage, Weights, Decode, Prefill, 2 + 2 * Config.Layers, Layout);
    Hashes.AddPair(ParameterFile, HashFile(BundleMember(Stage, ParameterFile)));
    CompileGraph(Stage, DecodeLibrary, Kit, Decode, Hashes);
    CompileGraph(Stage, PrefillLibrary, Kit, Prefill, Hashes);
    WriteManifest(Stage, Manifest);
    RequireCpu(MoveFileW(PWideChar(Stage), PWideChar(Output)), 'Could not install the Gemma 2 bundle');
    Published := True;
  finally
    Prefill.Free;
    Decode.Free;
    Manifest.Free;
    Weights.Free;
    Data.Free;
    if (Stage <> '') and not Published then RemoveStage(Stage);
  end;
end;

end.
