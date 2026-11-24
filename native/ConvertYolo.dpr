{ ************************************************************************** }
{                                                                            }
{ ConvertYolo                                                                }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertYolo;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}{$IFDEF UNIX}CThreads, CWString,{$ENDIF}SysUtils
  {$ELSE}System.SysUtils{$ENDIF}, CrossPascal.Json, CrossPascal.YoloBundle, CrossPascal.NativeIO;

var
  Receipt: TJSONObject;
  ValidateOnly: Boolean;

begin
  ValidateOnly := (NativeParamCount = 2) and (NativeParamStr(2) = '--validate');
  if (NativeParamCount <> 1) and not ValidateOnly then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertYolo request.json [--validate]');
    Halt(2);
  end;
  try
    Receipt := ConvertYoloBundle(NativeParamStr(1), ValidateOnly);
    try
      NativeWriteUTF8(Receipt.ToJSON + #10);
    finally
      Receipt.Free;
    end;
  except
    on E: Exception do
    begin
      NativeWriteUTF8(E.ClassName + ': ' + E.Message + #10, True);
      Halt(1);
    end;
  end;
end.
