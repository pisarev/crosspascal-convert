{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nVisionSource                                            }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nVisionSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.GGUF, CrossPascal.Onnx;

procedure WriteGemma3nVisionWeights(const Projector: TGGUFCheckpoint; const Graph: TOnnxModel;
  const Directory: string);

implementation

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF},
  CrossPascal.Json, TVM.CpuGraph;

const
  HiddenSize = 2048;
  ImageTokens = 256;
  HardTokens = 128;

procedure RequireElements(const Tensor: TGGUFTensor; const Shape: TArray<Int64>);
begin
  RequireCpu((Length(Shape) >= 1) and (Length(Shape) <= Length(Tensor.Dimensions)) and (Tensor.Elements = TensorElements(Shape)),
    'Gemma 3n projector tensor size differs: ' + Tensor.Name);
end;

function ConcreteShape(const Value: TOnnxValue): TArray<Int64>;
var
  I: Integer;
begin
  RequireCpu(Value.HasShape and (Value.DataType = 1), 'Expected a shaped FP32 vision value: ' + Value.Name);
  SetLength(Result, Length(Value.Shape));
  for I := 0 to High(Result) do Result[I] := Value.Shape[I].Value;
  TensorElements(Result);
end;

function SourceName(const GraphName: string): string;
const
  EncoderPrefix = 'encoder.';
begin
  if GraphName = 'norm' then Exit('mm.soft_emb_norm.weight');
  if GraphName = 'proj' then Exit('mm.input_projection.weight');
  RequireCpu(Copy(GraphName, 1, Length(EncoderPrefix)) = EncoderPrefix,
    'Unrecognized Gemma 3n vision parameter: ' + GraphName);
  Result := 'v.' +
    StringReplace(Copy(GraphName, Length(EncoderPrefix) + 1, MaxInt), 'blocks.', 'blk.', [rfReplaceAll]);
end;

procedure WriteNpy(const Projector: TGGUFCheckpoint; const Name, FileName: string;
  const Shape: TArray<Int64>);
const
  Magic: array[0..7] of Byte = ($93, Ord('N'), Ord('U'), Ord('M'), Ord('P'), Ord('Y'), 1, 0);
var
  FileStream: TFileStream;
  Text, Dims: UTF8String;
  HeaderSize: Word;
  I: Integer;
begin
  RequireElements(Projector.Tensor(Name), Shape);
  Dims := '';
  for I := 0 to High(Shape) do
  begin
    if I > 0 then Dims := Dims + ', ';
    Dims := Dims + UTF8Encode(IntToStr(Shape[I]));
  end;
  if Length(Shape) = 1 then Dims := Dims + ',';
  Text := '{''descr'': ''<f4'', ''fortran_order'': False, ''shape'': (' + Dims + '), }';
  Text := Text + UTF8String(StringOfChar(' ', 63 - (Length(Text) + 10) mod 64)) + #10;
  HeaderSize := Length(Text);
  FileStream := TFileStream.Create(FileName, fmCreate);
  try
    FileStream.WriteBuffer(Magic, SizeOf(Magic));
    FileStream.WriteBuffer(HeaderSize, SizeOf(HeaderSize));
    FileStream.WriteBuffer(Pointer(Text)^, Length(Text));
    Projector.WriteFloat32(Name, FileStream);
  finally
    FileStream.Free;
  end;
end;

procedure WriteGemma3nVisionWeights(const Projector: TGGUFCheckpoint; const Graph: TOnnxModel;
  const Directory: string);
var
  FileStream: TFileStream;
  Layout, Dimensions: TJSONArray;
  Row: TJSONObject;
  Shape: TArray<Int64>;
  Source: string;
  Text: UTF8String;
  Dim: Int64;
  Size: UInt32;
  I: Integer;
begin
  RequireCpu(Projector.Text('clip.vision.projector_type') = 'gemma3nv',
    'Expected a Gemma 3n vision projector');
  RequireCpu(
    (Length(Graph.Inputs) >= 3) and (Graph.Inputs[0].Name = 'pixels') and (Length(Graph.Outputs) = 1) and (Graph.Outputs[0].Name = 'embeddings'),
    'Invalid Gemma 3n vision graph signature'
  );
  Size := Projector.Unsigned('clip.vision.image_size');
  Shape := ConcreteShape(Graph.Inputs[0]);
  RequireCpu((Length(Shape) = 4) and (Shape[0] = 1) and (Shape[1] = 3) and (Shape[2] = Size) and (Shape[3] = Size),
    'Gemma 3n vision resolution differs from the projector');
  Shape := ConcreteShape(Graph.Outputs[0]);
  RequireCpu((Length(Shape) = 3) and (Shape[0] = 1) and (Shape[1] = ImageTokens) and (Shape[2] = HiddenSize),
    'Gemma 3n vision output dimensions differ from the runtime profile');
  for I := 1 to High(Graph.Inputs) do
    RequireElements(Projector.Tensor(SourceName(Graph.Inputs[I].Name)), ConcreteShape(Graph.Inputs[I]));
  Layout := TJSONArray.Create;
  try
    FileStream := TFileStream.Create(BundleMember(Directory, 'vision-params.bin'), fmCreate);
    try
      for I := 1 to High(Graph.Inputs) do
      begin
        Shape := ConcreteShape(Graph.Inputs[I]);
        Source := SourceName(Graph.Inputs[I].Name);
        Row := TJSONObject.Create;
        Layout.AddElement(Row);
        Row.AddPair('name', Graph.Inputs[I].Name);
        Row.AddPair('dtype', 'float32');
        Row.AddPair('offset', JSONInteger(FileStream.Position));
        Dimensions := TJSONArray.Create;
        Row.AddPair('shape', Dimensions);
        for Dim in Shape do Dimensions.AddElement(JSONInteger(Dim));
        Projector.WriteFloat32(Source, FileStream);
      end;
    finally
      FileStream.Free;
    end;
    {$IFDEF FPC}
    Text := UTF8String(Layout.AsJSON) + UTF8String(sLineBreak);
    {$ELSE}
    Text := UTF8Encode(Layout.ToJSON + sLineBreak);
    {$ENDIF}
    FileStream := TFileStream.Create(BundleMember(Directory, 'vision-parameters.json'), fmCreate);
    try
      FileStream.WriteBuffer(Pointer(Text)^, Length(Text));
    finally
      FileStream.Free;
    end;
  finally
    Layout.Free;
  end;
  WriteNpy(Projector, 'mm.embedding.weight', BundleMember(Directory, 'vision-hard.npy'),
    [HardTokens, HiddenSize]);
  WriteNpy(Projector, 'mm.hard_emb_norm.weight', BundleMember(Directory, 'vision-hard-norm.npy'),
    [HiddenSize]);
  WriteNpy(Projector, 'mm.input_projection.weight',
    BundleMember(Directory, 'vision-projection.npy'), [HiddenSize, HiddenSize]);
end;

end.
