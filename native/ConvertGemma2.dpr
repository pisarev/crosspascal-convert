{ ************************************************************************** }
{                                                                            }
{ ConvertGemma2                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev (ypisareff@outlook.com)                     }
{                                                                            }
{ ************************************************************************** }

program ConvertGemma2;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.Gemma2Bundle;

var
  Kit: TWindowsTvmKit;

begin
  if ParamCount <> 10 then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertGemma2 checkpoint-directory destination context prefill-chunk');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(5);
    Kit.Linker := ParamStr(6);
    Kit.FfiLibrary := ExpandFileName(ParamStr(7));
    Kit.MsvcLibraries := ParamStr(8);
    Kit.UcrtLibraries := ParamStr(9);
    Kit.WindowsLibraries := ParamStr(10);
    ConvertGemma2Bundle(ParamStr(1), ParamStr(2), StrToInt(ParamStr(3)), StrToInt(ParamStr(4)), Kit);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(2))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
