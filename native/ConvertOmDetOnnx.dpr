{ ************************************************************************** }
{                                                                            }
{ ConvertOmDetOnnx                                                           }
{                                                                            }
{ Copyright © 2026 Yuriy Pisarev                                             }
{                                                                            }
{ ************************************************************************** }

program ConvertOmDetOnnx;

{$APPTYPE CONSOLE}
{$IFDEF FPC}{$MODE DELPHIUNICODE}{$ENDIF}

uses
  {$IFDEF FPC}SysUtils{$ELSE}System.SysUtils{$ENDIF}, CrossPascal.TVMCompiler, CrossPascal.OmDetBundle,
  CrossPascal.NativeIO;

var
  Kit: TWindowsTvmKit;
  Nvrtc, Architecture: string;
  RequiredEnd: Integer;

begin
  RequiredEnd := NativeParamCount;
  if (NativeParamCount = 11) and (NativeParamStr(9) = '--cuda') then
  begin
    RequiredEnd := 8;
    Nvrtc := NativeParamStr(10);
    Architecture := NativeParamStr(11);
  end;
  if RequiredEnd <> 8 then
  begin
    WriteLn(ErrOutput, 'Usage: ConvertOmDetOnnx source-directory destination');
    WriteLn(ErrOutput, '       TVM-compiler-directory link.exe tvm_ffi.lib msvc-lib ucrt-lib windows-lib');
    WriteLn(ErrOutput, '       [--cuda nvrtc.dll sm_75|sm_86|sm_89|sm_120]');
    Halt(2);
  end;
  try
    Kit.CompilerDirectory := NativeParamStr(3);
    Kit.Linker := NativeParamStr(4);
    Kit.FfiLibrary := ExpandFileName(NativeParamStr(5));
    Kit.MsvcLibraries := NativeParamStr(6);
    Kit.UcrtLibraries := NativeParamStr(7);
    Kit.WindowsLibraries := NativeParamStr(8);
    ConvertOmDetBundle(NativeParamStr(1), NativeParamStr(2), Kit, Nvrtc, Architecture);
    WriteLn(UTF8Encode(ExpandFileName(NativeParamStr(2))));
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, UTF8Encode(E.Message));
      Halt(1);
    end;
  end;
end.
