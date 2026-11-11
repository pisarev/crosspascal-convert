{ ************************************************************************** }
{                                                                            }
{ CrossPascal.NativeBundle                                                   }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.NativeBundle;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json, CrossPascal.Onnx, CrossPascal.SafeTensors;

function BundleDimensions(const Shape: array of Int64): TJSONArray;
function BundleFileSHA256(const FileName: string): string;
procedure WriteBundleJSON(const FileName: string; const Manifest: TJSONObject);
procedure DeleteBundleStage(const Folder: string);
procedure WriteGraphWeights(const FileName: string; const Checkpoint: TSafeTensorCheckpoint;
  const Decode, Prefill: TOnnxModel; const RuntimeInputs: Integer; const Layout: TJSONArray);

implementation

uses
  {$IFDEF FPC}Windows, SysUtils, Classes{$ELSE}Winapi.Windows, System.SysUtils, System.Classes{$ENDIF},
  CrossPascal.SHA256, TVM.CpuGraph;

function BundleDimensions(const Shape: array of Int64): TJSONArray;
var
  Dimension: Int64;
begin
  Result := TJSONArray.Create;
  for Dimension in Shape do Result.AddElement(JSONInteger(Dimension));
end;

function BundleFileSHA256(const FileName: string): string;
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

procedure WriteBundleJSON(const FileName: string; const Manifest: TJSONObject);
var
  Stream: TFileStream;
  Text: UTF8String;
begin
  Text := UTF8Encode(Manifest.ToJSON + #10);
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    Stream.WriteBuffer(Text[1], Length(Text));
  finally
    Stream.Free;
  end;
end;

procedure DeleteBundleStage(const Folder: string);
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

procedure WriteGraphWeights(const FileName: string; const Checkpoint: TSafeTensorCheckpoint;
  const Decode, Prefill: TOnnxModel; const RuntimeInputs: Integer; const Layout: TJSONArray);
var
  Stream: TFileStream;
  Weight, Other: TOnnxValue;
  Shape: TArray<Int64>;
  Entry: TJSONObject;
  Offset, Expected: Int64;
  I, J: Integer;
begin
  RequireCpu(Length(Decode.Inputs) = Length(Prefill.Inputs), 'Graph input count changed');
  Stream := TFileStream.Create(FileName, fmCreate);
  try
    for I := RuntimeInputs to High(Decode.Inputs) do
    begin
      Weight := Decode.Inputs[I];
      Other := Prefill.Inputs[I];
      RequireCpu((Weight.Name = Other.Name) and (Length(Weight.Shape) = Length(Other.Shape)),
        'Graph weight binding order changed');
      SetLength(Shape, Length(Weight.Shape));
      for J := 0 to High(Shape) do
      begin
        Shape[J] := Weight.Shape[J].Value;
        RequireCpu((Shape[J] > 0) and (Shape[J] = Other.Shape[J].Value),
          'Graph weight shape changed: ' + Weight.Name);
      end;
      Offset := Stream.Position;
      Expected := TensorElements(Shape) * SizeOf(Single);
      Checkpoint.WriteFloat32(Weight.Name, Stream);
      RequireCpu(Stream.Position - Offset = Expected, 'Incorrect parameter size: ' + Weight.Name);
      Entry := TJSONObject.Create;
      Layout.AddElement(Entry);
      Entry.AddPair('name', Weight.Name);
      Entry.AddPair('offset', JSONInteger(Offset));
      Entry.AddPair('shape', BundleDimensions(Shape));
    end;
  finally
    Stream.Free;
  end;
end;

end.
