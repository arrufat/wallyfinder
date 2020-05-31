#include <iostream>

auto main(const int argc, const char** argv) -> int
try
{
    std::cout << "C++ Project Template\n";
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
