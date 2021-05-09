# libtorch-easy
A standalone minimal example of how to use libtorch (PyTorch C++ API).

### Download the pre-compiled libtorch library with cxx11-ABI and CUDA 11.1
```
mkdir $HOME/local && cd $HOME/local
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip
unzip libtorch*.zip

export LIBTORCH_SYS=$HOME/local/libtorch
```

### Compile an example program
```
g++ torch-example.cc \
-std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-I${LIBTORCH_SYS}/include \
-I${LIBTORCH_SYS}/include/torch/csrc/api/include \
-L${LIBTORCH_SYS}/lib \
-Wl,-R${LIBTORCH_SYS}/lib \
-ltorch -ltorch_cpu -lc10 -lgomp \
-o torch-example
```
</br>

_torch-example.cc_
```
#include <iostream>
#include <iomanip>
#include <torch/torch.h>

int main() {
    
    const int64_t input_size   = 1;
    const int64_t output_size  = 1;
    
    const double learning_rate = 0.01;
    const size_t num_epochs    = 150;
    
    // Random dataset
    auto x_train = torch::randint(0, 5, {20, 1});
    auto y_train = torch::randint(0, 5, {20, 1});
    
    // Linear regression
    torch::nn::Linear model(input_size, output_size);
    
    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    
    std::cout << std::fixed << std::setprecision(5);
    
    // Train loop
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        
        // Forward
        auto output = model(x_train);
        auto loss = torch::nn::functional::mse_loss(output, y_train);

        // Backward and gradient step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        std::cout << "Epoch: " << epoch + 1 << "/" << num_epochs
                  << " | Loss: " << loss.item<double>() << std::endl;
    }
    return 0;
}
```

For code tutorials of models implemented in C++, see: https://github.com/prabhuomkar/pytorch-cpp

</br>

m.mieskolainen@imperial.ac.uk, 2021
