{ ************************************************************************** }
{                                                                            }
{ CrossPascal.YoloSource                                                     }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.YoloSource;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Onnx;

function BindYoloCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string): TOnnxModel;

implementation

uses
  {$IFDEF FPC}SysUtils, Math{$ELSE}System.SysUtils, System.Math{$ENDIF},
  CrossPascal.TorchCheckpoint, TVM.CpuGraph;

function ReadWeight(const Source: TTorchCheckpoint; const Name: string;
  const ExpectedShape: array of Int64): TArray<Single>;
var
  Shape: TArray<Int64>;
  I: Integer;
begin
  Result := Source.ReadFloats(Name, Shape);
  RequireCpu(Length(Shape) = Length(ExpectedShape), 'Wrong PyTorch tensor rank: ' + Name);
  for I := 0 to High(Shape) do
    RequireCpu(Shape[I] = ExpectedShape[I], 'Wrong PyTorch tensor dimensions: ' + Name);
end;

function IsSuffix(const Name, Suffix: string): Boolean;
begin
  Result := Copy(Name, Length(Name) - Length(Suffix) + 1, MaxInt) = Suffix;
end;

function FoldBatchNorm(const Source: TTorchCheckpoint; const Tensor: TOnnxTensor;
  const BatchNorm: string; const IsBias: Boolean): TArray<Single>;
var
  Gamma, Variance, Mean, Beta, ConvBias: TArray<Single>;
  Scale, Sigma, Epsilon: Single;
  Channels, RowWidth, Channel, I, At: Integer;
begin
  Channels := Tensor.Shape[0];
  Gamma := ReadWeight(Source, BatchNorm + '.weight', [Channels]);
  Variance := ReadWeight(Source, BatchNorm + '.running_var', [Channels]);
  Epsilon := Source.BatchNormEpsilon(BatchNorm);
  if IsBias then
  begin
    Mean := ReadWeight(Source, BatchNorm + '.running_mean', [Channels]);
    Beta := ReadWeight(Source, BatchNorm + '.bias', [Channels]);
    if Source.Contains(Tensor.Name) then
      ConvBias := ReadWeight(Source, Tensor.Name, [Channels])
    else
      SetLength(ConvBias, Channels);
    SetLength(Result, Channels);
    for Channel := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[Channel] + Epsilon);
      Scale := Gamma[Channel] / Sigma;
      Result[Channel] := Scale * ConvBias[Channel] +
        (Beta[Channel] - Gamma[Channel] * Mean[Channel] / Sigma);
    end;
  end
  else begin
    Result := ReadWeight(Source, Tensor.Name, Tensor.Shape);
    RowWidth := Length(Result) div Channels;
    At := 0;
    for Channel := 0 to Channels - 1 do
    begin
      Sigma := Sqrt(Variance[Channel] + Epsilon);
      Scale := Gamma[Channel] / Sigma;
      for I := 1 to RowWidth do
      begin
        Result[At] := Result[At] * Scale;
        Inc(At);
      end;
    end;
  end;
end;

function BindYoloCheckpoint(const Recipe: TOnnxModel; const CheckpointFile: string): TOnnxModel;
const
  ConvWeight = '.conv.weight';
  ConvBias = '.conv.bias';
var
  Source: TTorchCheckpoint;
  Initializers: TArray<TOnnxTensor>;
  Buffers: TArray<TBytes>;
  Values: TArray<Single>;
  Name, BatchNorm: string;
  I, J, Replaced: Integer;
  IsBias: Boolean;
begin
  Source := TTorchCheckpoint.Create(CheckpointFile);
  try
    Initializers := Copy(Recipe.Initializers);
    SetLength(Buffers, Length(Initializers));
    Replaced := 0;
    for I := 0 to High(Initializers) do
    begin
      Name := Initializers[I].Name;
      if Copy(Name, 1, 6) <> 'model.' then Continue;
      RequireCpu((Initializers[I].DataType = 1) and (Length(Initializers[I].Shape) > 0) and
        (Initializers[I].Shape[0] > 0) and (Initializers[I].RawData.Size = 0) and
        not Initializers[I].IsExternal, 'Invalid YOLO weight placeholder: ' + Name);
      IsBias := IsSuffix(Name, ConvBias);
      BatchNorm := '';
      if IsBias then BatchNorm := Copy(Name, 1, Length(Name) - Length(ConvBias)) + '.bn';
      if IsSuffix(Name, ConvWeight) then
        BatchNorm := Copy(Name, 1, Length(Name) - Length(ConvWeight)) + '.bn';
      if (BatchNorm <> '') and Source.Contains(BatchNorm + '.running_var') then
        Values := FoldBatchNorm(Source, Initializers[I], BatchNorm, IsBias)
      else
        Values := ReadWeight(Source, Name, Initializers[I].Shape);
      for J := 0 to High(Values) do
        RequireCpu(not IsNan(Values[J]) and not IsInfinite(Values[J]), 'Nonfinite YOLO weight: ' + Name);
      SetLength(Buffers[I], Length(Values) * SizeOf(Single));
      Move(Values[0], Buffers[I][0], Length(Buffers[I]));
      Initializers[I].Serialized := Default(TOnnxBytes);
      Initializers[I].RawData.Data := @Buffers[I][0];
      Initializers[I].RawData.Size := Length(Buffers[I]);
      Inc(Replaced);
    end;
    RequireCpu(Replaced > 0, 'YOLO graph recipe does not contain weight placeholders');
    Result := TOnnxModel.CreateGenerated(Recipe.Opset, Recipe.Inputs, Recipe.Outputs, Initializers,
      Recipe.Nodes, Buffers);
  finally
    Source.Free;
  end;
end;

end.
