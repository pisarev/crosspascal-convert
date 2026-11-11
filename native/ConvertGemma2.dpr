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
  NvrtcLibrary, Architecture: string;

begin
  if (ParamCount <> 10) and not ((ParamCount = 13) and (ParamStr(11) = '--cuda')) then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertGemma2 checkpoint-directory destination context prefill-chunk');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_XX]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := ParamStr(5);
    Kit.Linker := ParamStr(6);
    Kit.FfiLibrary := ExpandFileName(ParamStr(7));
    Kit.MsvcLibraries := ParamStr(8);
    Kit.UcrtLibraries := ParamStr(9);
    Kit.WindowsLibraries := ParamStr(10);
    if ParamCount = 13 then
    begin
      NvrtcLibrary := ParamStr(12);
      Architecture := ParamStr(13);
    end;
    ConvertGemma2Bundle(ParamStr(1), ParamStr(2), StrToInt(ParamStr(3)), StrToInt(ParamStr(4)),
      Kit, NvrtcLibrary, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(ParamStr(2))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
