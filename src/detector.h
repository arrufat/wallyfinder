#ifndef detector_h_INCLUDED
#define detector_h_INCLUDED

#include <dlib/dnn.h>
#include <dlib/base64.h>
#include <dlib/compress_stream.h>

// HOW TO GENERATE THE COMPRESSED STREAM:
/*
    dlib::deserialize(model_path) >> net;
    std::ostringstream sout;
    std::istringstream sin;
    dlib::base64 base64_coder;
    dlib::compress_stream::kernel_2a compressor;
    // put the data into ostream sout
    dlib::serialize(net, sout);
    // put the data into istream sin
    sin.str(sout.str());
    sout.str("");
    // compress the data
    compressor.compress(sin, sout);
    sin.clear();
    sin.str(sout.str());
    sout.str("");
    // encode the data into base64
    base64_coder.encode(sin, sout);
    // print the data on the terminal
    std::cout << sout.str() << std::endl;
*/

namespace detector
{
    using namespace dlib;

    template <long num_filters, int stride, template <typename> class BN, typename SUBNET>
    using rcon3 = relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>;

    // clang-format off
    template <template <typename> class BN>
    using net_type = loss_mmod<con<1, 3, 3, 1, 1,
    rcon3<55, 1, BN,
    rcon3<55, 1, BN,
    rcon3<55, 1, BN,
    rcon3<32, 2, BN,
    rcon3<32, 2, BN,
    rcon3<16, 2, BN,
    input_rgb_image_pyramid<pyramid_down<6>>
    >>>>>>>>;
    // clang-format on

    using train = net_type<bn_con>;
    using infer = net_type<affine>;

    const std::string load();

}  // namespace detector

#endif  // detector_h_INCLUDED
