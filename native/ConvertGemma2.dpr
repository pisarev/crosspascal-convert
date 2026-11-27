{ ************************************************************************** }
{                                                                            }
{ ConvertGemma2                                                              }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertGemma2;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.Gemma2Bundle,
  CrossPascal.NativeIO;

var
  Kit: TWindowsTvmKit;
  NvrtcLibrary, Architecture: string;

begin
  if (NativeParamCount <> 10) and not ((NativeParamCount = 13) and (NativeParamStr(11) = '--cuda')) then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertGemma2 checkpoint-directory destination context prefill-chunk');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_XX]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := NativeParamStr(5);
    Kit.Linker := NativeParamStr(6);
    Kit.FfiLibrary := ExpandFileName(NativeParamStr(7));
    Kit.MsvcLibraries := NativeParamStr(8);
    Kit.UcrtLibraries := NativeParamStr(9);
    Kit.WindowsLibraries := NativeParamStr(10);
    if NativeParamCount = 13 then
    begin
      NvrtcLibrary := NativeParamStr(12);
      Architecture := NativeParamStr(13);
    end;
    ConvertGemma2Bundle(NativeParamStr(1), NativeParamStr(2), StrToInt(NativeParamStr(3)),
      StrToInt(NativeParamStr(4)), Kit, NvrtcLibrary, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(NativeParamStr(2))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
