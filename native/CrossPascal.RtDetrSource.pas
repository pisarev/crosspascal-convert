{ ************************************************************************** }
{                                                                            }
{ CrossPascal.RtDetrSource                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.RtDetrSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx;

function BindRtDetrCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string;
  const Bindings: TJSONObject): TOnnxModel;

implementation

uses
  {$IFDEF FPC}Classes, Math, SysUtils{$ELSE}System.Classes, System.Math, System.SysUtils{$ENDIF},
  CrossPascal.SafeTensors, TVM.CpuGraph;

function ReadWeight(const Source: TSafeTensorFile; const Name: string;
  const Shape: array of Int64): TArray<Single>;
var
  Info: TSafeTensorInfo;
  Buffer: TMemoryStream;
  I: Integer;
begin
  Info := Source.TensorInfo(Name);
  RequireCpu(
    (Length(Info.Shape) = Length(Shape)) and (Info.ElementCount <= MaxInt div SizeOf(Single)) and ((Info.DType = 'F32') or (Info.DType = 'F16') or (Info.DType = 'BF16')),
    'Invalid RT-DETR tensor: ' + Name
  );
  for I := 0 to High(Shape) do
    RequireCpu(Shape[I] = Info.Shape[I], 'RT-DETR tensor shape differs: ' + Name);
  Buffer := TMemoryStream.Create;
  try
    Source.WriteFloat32(Name, Buffer);
    SetLength(Result, Info.ElementCount);
    RequireCpu(Buffer.Size = Int64(Length(Result)) * SizeOf(Single),
      'RT-DETR weight byte count differs: ' + Name);
    Move(Buffer.Memory^, Result[0], Buffer.Size);
  finally
    Buffer.Free;
  end;
end;

function ReadBinding(const Source: TSafeTensorFile; const Binding: TJSONObject;
  const Shape: TArray<Int64>; const Epsilon: Single): TArray<Single>;
var
  Name, Norm, Mode: string;
  Weights, Gamma, Variance, Mean, Beta: TArray<Single>;
  Channels, Cols, Rows, Row, Col, I, At: Integer;
  Sigma, Scale: Single;
begin
  Name := ModelText(Binding.GetValue('source'));
  Mode := ModelText(Binding.GetValue('mode'));
  if Mode = 'copy' then Exit(ReadWeight(Source, Name, Shape));
  if Mode = 'transpose' then
  begin
    RequireCpu(Length(Shape) = 2, 'Invalid RT-DETR transpose binding');
    Rows := Shape[0];
    Cols := Shape[1];
    Weights := ReadWeight(Source, Name, [Cols, Rows]);
    SetLength(Result, Length(Weights));
    for Row := 0 to Rows - 1 do
      for Col := 0 to Cols - 1 do Result[Row * Cols + Col] := Weights[Col * Rows + Row];
    Exit;
  end;
  Norm := ModelText(Binding.GetValue('normalization'));
  RequireCpu(Norm <> '', 'RT-DETR normalization binding is missing');
  if Mode = 'conv_weight' then
  begin
    RequireCpu(Length(Shape) = 4, 'Invalid RT-DETR convolution weight binding');
    Channels := Shape[0];
  end
  else if Mode = 'conv_bias' then
    begin
      RequireCpu(Length(Shape) = 1, 'Invalid RT-DETR convolution bias binding');
      Channels := Shape[0];
    end
    else begin
      RequireCpu(((Mode = 'scale') or (Mode = 'bias')) and (Length(Shape) = 4) and (Shape[0] = 1) and (Shape[2] = 1) and (Shape[3] = 1),
        'Invalid RT-DETR normalization binding');
      Channels := Shape[1];
    end;
  Gamma := ReadWeight(Source, Norm + '.weight', [Channels]);
  Variance := ReadWeight(Source, Norm + '.running_var', [Channels]);
  if Mode = 'conv_weight' then
  begin
    Result := ReadWeight(Source, Name, Shape);
    Cols := Length(Result) div Channels;
    At := 0;
    for Row := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[Row] + Epsilon);
      Scale := Gamma[Row] / Sigma;
      for Col := 1 to Cols do
      begin
        Result[At] := Result[At] * Scale;
        Inc(At);
      end;
    end;
  end
  else begin
    if Mode <> 'scale' then
    begin
      Mean := ReadWeight(Source, Norm + '.running_mean', [Channels]);
      Beta := ReadWeight(Source, Norm + '.bias', [Channels]);
    end;
    SetLength(Result, Channels);
    for I := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[I] + Epsilon);
      Scale := Gamma[I] / Sigma;
      if Mode = 'scale' then
        Result[I] := Scale
      else
        Result[I] := Beta[I] - Mean[I] * Scale;
    end;
  end;
end;

function BindRtDetrCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string;
  const Bindings: TJSONObject): TOnnxModel;
var
  Source: TSafeTensorFile;
  Rows: TJSONObject;
  Initializers: TArray<TOnnxTensor>;
  Buffers: TArray<TBytes>;
  Values: TArray<Single>;
  Epsilon: Single;
  I, J: Integer;
begin
  Rows := ModelObject(Bindings.GetValue('tensors'), 'RT-DETR tensor bindings');
  Epsilon := ModelFloat(Bindings.GetValue('epsilon'), 'RT-DETR normalization epsilon');
  RequireCpu((Epsilon > 0) and (Epsilon < 1) and (Rows.Count = Length(Recipe.Initializers)),
    'RT-DETR recipe bindings differ from the graph');
  Source := TSafeTensorFile.Create(CheckpointFile);
  try
    Initializers := Copy(Recipe.Initializers);
    SetLength(Buffers, Length(Initializers));
    for I := 0 to High(Initializers) do
    begin
      RequireCpu((Initializers[I].DataType = 1) and (Length(Initializers[I].Shape) > 0) and
        (Initializers[I].RawData.Size = 0) and not Initializers[I].IsExternal,
        'Invalid RT-DETR weight placeholder: ' + Initializers[I].Name);
      Values := ReadBinding(Source,
        ModelObject(Rows.GetValue(Initializers[I].Name), Initializers[I].Name),
        Initializers[I].Shape, Epsilon);
      for J := 0 to High(Values) do
        RequireCpu(not IsNan(Values[J]) and not IsInfinite(Values[J]),
          'Nonfinite RT-DETR weight: ' + Initializers[I].Name);
      SetLength(Buffers[I], Length(Values) * SizeOf(Single));
      Move(Values[0], Buffers[I][0], Length(Buffers[I]));
      Initializers[I].Serialized := Default(TOnnxBytes);
      Initializers[I].RawData.Data := @Buffers[I][0];
      Initializers[I].RawData.Size := Length(Buffers[I]);
    end;
    Result := TOnnxModel.CreateGenerated(Recipe.Opset, Recipe.Inputs, Recipe.Outputs, Initializers,
      Recipe.Nodes, Buffers);
  finally
    Source.Free;
  end;
end;

end.
