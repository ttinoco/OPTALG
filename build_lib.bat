IF "%OPTALG_IPOPT%" == "true" (
    IF NOT EXIST "lib\ipopt" (
	mkdir lib
        cd lib
        bitsadmin /transfer "JobName" https://www.coin-or.org/download/binary/Ipopt/Ipopt-3.11.1-win64-intel13.1.zip "%cd%\lib\ipopt.zip"
        7z x ipopt.zip
        for /d %%G in ("Ipopt*") do move "%%~G" ipopt
        cd ipopt
      	mkdir temp
        cd temp
        bitsadmin /transfer "JobName" https://www.coin-or.org/download/binary/Ipopt/Ipopt-3.11.0-Win32-Win64-dll.7z "%cd%\lib\ipopt\temp\foo.7z" 
        7z x foo.7z
        copy lib\x64\ReleaseMKL\IpOptFSS.dll ..\lib
        copy lib\x64\ReleaseMKL\IpOpt-vc10.dll ..\lib
        cd ..
        copy lib\Ipopt-vc10.dll ..\..\optalg\opt_solver\_ipopt
	copy lib\IpoptFSS.dll ..\..\optalg\opt_solver\_ipopt
        copy lib\IpoptFSS.dll ..\..\optalg\lin_solver\_mumps
        cd ..\..\
        python setup.py setopt --command build -o compiler -s mingw32
    )
) 

