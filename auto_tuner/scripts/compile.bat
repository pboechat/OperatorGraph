@echo off
setlocal
cd %1
call "%VS120COMNTOOLS%"\vsvars32.bat
rem NOTE: Debug configurations most probably hang standard output (ie.: std::getchar()) and won't work with the optimization pipeline!
MSBuild /m /t:%2 /p:Platform=x64 /p:Configuration=Release build\vs2013\partition_test.vcxproj > compile.log 2>&1
exit /b %errorlevel%
endlocal