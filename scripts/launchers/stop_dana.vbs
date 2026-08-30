' Silent headless shutdown — no console flash.
' Resolves sibling stop_dana.bat via ScriptFullName (VBS has no %~dp0).
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c """ & dir & "\stop_dana.bat""", 0, False
