{ ************************************************************************** }
{                                                                            }
{ CrossPascal.Gemma3nDistribution                                             }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                              }
{                                                                            }
{ ************************************************************************** }

unit CrossPascal.Gemma3nDistribution;

{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

interface

uses
  CrossPascal.Json;

procedure CheckGemma3nDistribution(const Profile, SourceSHA256: string; const Required: Boolean);
procedure AddGemma3nDistribution(const Profile, Destination, SourceSHA256: string;
  const Manifest: TJSONObject; const Required: Boolean);

implementation

uses
  {$IFDEF FPC}Classes, SysUtils, Windows{$ELSE}System.Classes, System.SysUtils, Winapi.Windows{$ENDIF},
  CrossPascal.SHA256, CrossPascal.TokenizerJson, TVM.CpuGraph;

const
  DistributionFile = 'distribution.json';
  Members: array[0..4] of string = ('LICENSE-model.txt', 'USE-POLICY-model.txt',
    'NOTICE.txt', 'MODIFICATIONS-model.txt', 'SOURCE-model.json');
  Fields: array[0..4] of string = ('license_file', 'usage_policy', 'notice_file',
    'model_modifications', 'source_attribution');
  SourceFields: array[0..2] of string = ('license', 'source_repository', 'source_revision');

function FileHash(const Path: string): string;
var
  Stream: THandleStream;
  Handle: THandle;
begin
  Handle := CreateFileW(PWideChar(Path), GENERIC_READ, FILE_SHARE_READ, nil,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if Handle = INVALID_HANDLE_VALUE then RaiseLastOSError;
  try
    Stream := THandleStream.Create(Handle);
    try
      Result := SHA256Stream(Stream);
    finally
      Stream.Free;
    end;
  finally
    CloseHandle(Handle);
  end;
end;

function PathExists(const Path: string): Boolean;
var
  Error: DWORD;
begin
  Result := GetFileAttributesW(PWideChar(Path)) <> INVALID_FILE_ATTRIBUTES;
  if not Result then
  begin
    Error := GetLastError;
    if (Error <> ERROR_FILE_NOT_FOUND) and (Error <> ERROR_PATH_NOT_FOUND) then
      RaiseLastOSError(Error);
  end;
end;

function IsHex(const Value: string; const Count: Integer): Boolean;
var
  C: Char;
begin
  Result := Length(Value) = Count;
  if not Result then Exit;
  for C in Value do
    if not CharInSet(C, ['0'..'9', 'a'..'f', 'A'..'F']) then Exit(False);
end;

function ReadDistribution(const Profile, SourceSHA256: string;
  const Required: Boolean): TJSONObject;
var
  Data: TJSONValue;
  Files: TJSONObject;
  Reader: TUnicodeJsonReader;
  Name, Expected: string;
begin
  Result := nil;
  if not PathExists(BundleMember(Profile, DistributionFile)) then
  begin
    RequireCpu(not Required, 'The Gemma 3n profile requires distribution.json');
    Exit;
  end;
  Reader := TUnicodeJsonReader.Create(BundleMember(Profile, DistributionFile));
  try
    Data := ParseCrossJSON(Reader.Text);
  finally
    Reader.Free;
  end;
  try
    Result := ModelObject(Data, 'Gemma 3n distribution');
    RequireCpu(ModelText(Result.GetValue('schema')) = 'crosspascal.model-distribution.v1',
      'Unsupported Gemma 3n distribution descriptor');
    Expected := ModelText(Result.GetValue('source_sha256'));
    RequireCpu(IsHex(Expected, 64) and SameText(Expected, SourceSHA256),
      'Gemma 3n distribution source SHA256 differs from the checkpoint');
    RequireCpu((ModelText(Result.GetValue('license')) = 'gemma') and
      (ModelText(Result.GetValue('source_repository')) <> '') and
      IsHex(ModelText(Result.GetValue('source_revision')), 40),
      'Incomplete Gemma 3n distribution attribution');
    Files := ModelObject(Result.GetValue('files'), 'Gemma 3n distribution files');
    RequireCpu(Files.Count = Length(Members), 'Unexpected Gemma 3n distribution file count');
    for Name in Members do
    begin
      Expected := ModelText(Files.GetValue(Name));
      RequireCpu(IsHex(Expected, 64) and
        SameText(FileHash(BundleMember(Profile, Name)), Expected),
        'Gemma 3n distribution file changed: ' + Name);
    end;
    Data := nil;
  finally
    Data.Free;
  end;
end;

procedure CheckGemma3nDistribution(const Profile, SourceSHA256: string; const Required: Boolean);
var
  Distribution: TJSONObject;
begin
  Distribution := ReadDistribution(Profile, SourceSHA256, Required);
  Distribution.Free;
end;

procedure AddGemma3nDistribution(const Profile, Destination, SourceSHA256: string;
  const Manifest: TJSONObject; const Required: Boolean);
var
  Distribution, Files, Hashes: TJSONObject;
  I: Integer;
  Name, Expected, Target: string;
begin
  Distribution := ReadDistribution(Profile, SourceSHA256, Required);
  if Distribution = nil then Exit;
  try
    RequireCpu((ModelText(Manifest.GetValue('schema')) = 'crosspascal.gemma3n.text.v1') and
      SameText(ModelText(Manifest.GetValue('source_sha256')), SourceSHA256),
      'Gemma 3n distribution does not match the model manifest');
    Files := ModelObject(Distribution.GetValue('files'), 'Gemma 3n distribution files');
    Hashes := ModelObject(Manifest.GetValue('artifact_sha256'), 'Gemma 3n artifact hashes');
    for Name in SourceFields do
      RequireCpu(Manifest.GetValue(Name) = nil, 'Gemma 3n attribution already exists: ' + Name);
    for I := Low(Members) to High(Members) do
    begin
      Name := Members[I];
      RequireCpu((Manifest.GetValue(Fields[I]) = nil) and (Hashes.GetValue(Name) = nil) and
        not PathExists(BundleMember(Destination, Name)),
        'Gemma 3n distribution destination already exists: ' + Name);
    end;
    for I := Low(Members) to High(Members) do
    begin
      Name := Members[I];
      Target := BundleMember(Destination, Name);
      Expected := ModelText(Files.GetValue(Name));
      if not CopyFileW(PWideChar(BundleMember(Profile, Name)), PWideChar(Target), True) then
        RaiseLastOSError;
      RequireCpu(SameText(FileHash(Target), Expected),
        'Gemma 3n distribution copy changed: ' + Name);
      Hashes.AddPair(Name, Expected);
      Manifest.AddPair(Fields[I], Name);
    end;
    for Name in SourceFields do
      Manifest.AddPair(Name, ModelText(Distribution.GetValue(Name)));
  finally
    Distribution.Free;
  end;
end;

end.
