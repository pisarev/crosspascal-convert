{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nParameters                                               }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nParameters;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json;

function SelectGemma3nCudaParameters(const RecordData, Manifest: TJSONObject;
  out UnusedFiles: TArray<string>): TJSONObject;

implementation

uses
  {$IFDEF FPC}Classes, SysUtils{$ELSE}System.Classes, System.SysUtils{$ENDIF}, TVM.CpuGraph;

function CopyValue(const Value: TJSONValue): TJSONValue;
begin
  {$IFDEF FPC}
  Result := ParseCrossJSON(UTF8Decode(Value.AsJSON));
  {$ELSE}
  Result := ParseCrossJSON(Value.ToJSON);
  {$ENDIF}
  RequireCpu(Result <> nil, 'Could not copy a Gemma 3n parameter value');
end;

function SelectGemma3nCudaParameters(const RecordData, Manifest: TJSONObject;
  out UnusedFiles: TArray<string>): TJSONObject;
const
  Tables: array[0..1] of string = ('embed_tokens.weight', 'embed_tokens_per_layer.weight');
var
  Wanted, RetainedFiles, Excluded: TStringList;
  OriginalWeights, OriginalBindings, GraphBindings, KeptWeights, KeptBindings, Row: TJSONObject;
  Name, Tensor, FileName: string;
  I: Integer;
begin
  Result := nil;
  UnusedFiles := nil;
  RequireCpu((RecordData <> nil) and (Manifest <> nil), 'Gemma 3n parameter records are required');
  RequireCpu((ModelText(Manifest.GetValue('backend')) = 'TVM CUDA W8A32') and
    JSONIsTrue(Manifest.GetValue('host_embeddings')), 'Expected a Gemma 3n CUDA parameter record');
  Wanted := TStringList.Create;
  RetainedFiles := TStringList.Create;
  Excluded := TStringList.Create;
  try
    Wanted.CaseSensitive := True;
    Wanted.Sorted := True;
    Wanted.Duplicates := dupIgnore;
    RetainedFiles.CaseSensitive := False;
    RetainedFiles.Sorted := True;
    RetainedFiles.Duplicates := dupIgnore;
    Excluded.CaseSensitive := False;
    Excluded.Sorted := True;
    Excluded.Duplicates := dupIgnore;
    OriginalWeights := ModelObject(RecordData.GetValue('weights'), 'original weights');
    OriginalBindings := ModelObject(RecordData.GetValue('parameters'), 'original bindings');
    GraphBindings := ModelObject(Manifest.GetValue('parameters'), 'runtime parameters');
    for I := 0 to GraphBindings.Count - 1 do
    begin
      Row := ModelObject(JSONObjectValue(GraphBindings, I), 'runtime binding');
      Wanted.Add(ModelText(Row.GetValue('tensor')));
    end;
    for Name in Tables do
    begin
      Row := ModelObject(OriginalBindings.GetValue(Name), Name);
      Wanted.Add(ModelText(Row.GetValue('tensor')));
    end;
    for I := 0 to Wanted.Count - 1 do
      RequireCpu(OriginalWeights.GetValue(Wanted[I]) is TJSONObject,
        'Missing required Gemma 3n original tensor: ' + Wanted[I]);
    Result := TJSONObject.Create;
    try
      for I := 0 to RecordData.Count - 1 do
      begin
        Name := JSONObjectName(RecordData, I);
        if (Name <> 'weights') and (Name <> 'parameters') then
          Result.AddPair(Name, CopyValue(JSONObjectValue(RecordData, I)));
      end;
      KeptWeights := TJSONObject.Create;
      Result.AddPair('weights', KeptWeights);
      for I := 0 to OriginalWeights.Count - 1 do
      begin
        Name := JSONObjectName(OriginalWeights, I);
        Row := ModelObject(JSONObjectValue(OriginalWeights, I), Name);
        FileName := ModelText(Row.GetValue('file'));
        if Wanted.IndexOf(Name) >= 0 then
        begin
          KeptWeights.AddPair(Name, CopyValue(Row));
          RetainedFiles.Add(FileName);
        end
        else
          Excluded.Add(FileName);
      end;
      KeptBindings := TJSONObject.Create;
      Result.AddPair('parameters', KeptBindings);
      for I := 0 to OriginalBindings.Count - 1 do
      begin
        Row := ModelObject(JSONObjectValue(OriginalBindings, I), 'source binding');
        Tensor := ModelText(Row.GetValue('tensor'));
        if Wanted.IndexOf(Tensor) >= 0 then
          KeptBindings.AddPair(JSONObjectName(OriginalBindings, I), CopyValue(Row));
      end;
      for I := Excluded.Count - 1 downto 0 do
        if RetainedFiles.IndexOf(Excluded[I]) >= 0 then Excluded.Delete(I);
      SetLength(UnusedFiles, Excluded.Count);
      for I := 0 to Excluded.Count - 1 do UnusedFiles[I] := Excluded[I];
    except
      FreeAndNil(Result);
      raise;
    end;
  finally
    Excluded.Free;
    RetainedFiles.Free;
    Wanted.Free;
  end;
end;

end.
