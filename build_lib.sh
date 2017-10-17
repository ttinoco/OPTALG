#! /bin/sh

# IPOPT and MUMPS
if [ ! -d "lib/ipopt" ] && [ "$OPTALG_IPOPT" = true ]; then
    mkdir -p lib
    cd lib
    wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.8.zip
    unzip Ipopt-3.12.8.zip
    mv Ipopt-3.12.8 ipopt
    cd ipopt/ThirdParty/Mumps
    ./get.Mumps
    cd ../../
    ./configure
    make clean
    make uninstall
    make
    make install
    if [ "$(uname)" == "Darwin" ]; then
      install_name_tool -id "@rpath/libcoinmumps.1.dylib" lib/libcoinmumps.1.dylib
      install_name_tool -id "@rpath/libipopt.1.dylib" lib/libipopt.1.dylib
      install_name_tool -change "$PWD/lib/libcoinmumps.1.dylib" "@loader_path/../../lin_solver/_mumps/libcoinmumps.1.dylib" lib/libipopt.1.dylib
    fi
    cp lib/libipopt* ../../optalg/opt_solver/_ipopt
    cp lib/libcoinmumps* ../../optalg/lin_solver/_mumps
    cd ../../
fi

# CLP
if [ ! -d "lib/clp" ] && [ "$OPTALG_CLP" = true ]; then
    mkdir -p lib
    cd lib
    wget https://www.coin-or.org/download/source/Clp/Clp-1.16.11.zip 
    unzip Clp-1.16.11.zip
    mv Clp-1.16.11 clp
    cd clp
    ./configure
    make clean
    make uninstall
    make
    make install
    if [ "$(uname)" == "Darwin" ]; then
      install_name_tool -id "@rpath/libClp.1.dylib" lib/libClp.1.dylib
    fi
    cp lib/libClp* ../../optalg/opt_solver/_clp
    cd ../../
fi
