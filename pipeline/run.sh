if [ "$(basename "$(pwd)")" = "build" ]; then
    rm -rf *
    cmake -DCMAKE_PREFIX_PATH=/home/shaoze/Documents/Boeing/Boeing-Trajectory-Prediction/pipeline/libtorch ..
    cmake --build . --config Release
else
    echo "The current directory is not named 'build'."
fi
       