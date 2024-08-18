if [ "$(basename "$(pwd)")" = "build" ]; then
    rm -rf *
    current_dir=$(pwd)
    libtorch_absolute_path=$(realpath "$current_dir/../libtorch")
    cmake -DCMAKE_PREFIX_PATH=$libtorch_absolute_path ..
    cmake --build . --config Release -j8
else
    echo "The current directory is not named 'build'."
fi
       