#include "detector.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// convert a numpy array into a dlib::matrix<dlib::rgb_pixel>
const dlib::matrix<dlib::rgb_pixel> from_numpy(const py::array_t<unsigned char>& ndarray)
{
    const int num_channels = 3;
    auto raw = ndarray.unchecked<num_channels>();
    std::vector<unsigned char> rgb(3);
    dlib::matrix<dlib::rgb_pixel> image(raw.shape(0), raw.shape(1));
    for (ssize_t i = 0; i < raw.shape(0); i++)
    {
        for (ssize_t j = 0; j < raw.shape(1); j++)
        {
            image(i, j) = dlib::rgb_pixel(
                static_cast<unsigned char>(raw(i, j, 0)),
                static_cast<unsigned char>(raw(i, j, 1)),
                static_cast<unsigned char>(raw(i, j, 2)));
        }
    }
    return image;
}

// convert a dlib::matrix<dlib::rgb_pixel> into a numpy array
const py::array_t<unsigned char> to_numpy(const dlib::matrix<dlib::rgb_pixel>& image)
{
    const int num_channels = 3;
    auto data = std::make_unique<unsigned char[]>(image.size() * num_channels);
    int i = 0;
    for (const auto& pixel : image)
    {
        data[i++] = pixel.red;
        data[i++] = pixel.green;
        data[i++] = pixel.blue;
    }
    auto shape = std::vector<ptrdiff_t>{image.nr(), image.nc(), num_channels};
    return py::array_t<unsigned char>(shape, data.get());
}

auto sigmoid(const double val, const double alpha = 1.0) -> double
{
    return 1.0 / (1.0 + std::exp(-alpha * val));
}

auto logit(const double val, const double alpha = 1.0) -> double
{
    return 1.0 / alpha * (std::log(val) - std::log(1 - val));
}

class WallyFinder
{
    public:
    WallyFinder()
    {
        std::istringstream sin(detector::load());
        deserialize(net, sin);
    }

    auto clean() -> void { net.clean(); }

    auto forward(
        const py::array_t<unsigned char>& ndarray,
        const double confidence = 0.5,
        const size_t max_size = std::numeric_limits<size_t>::max()) -> const py::list
    {
        auto image = from_numpy(ndarray);
        auto scale = std::sqrt(static_cast<double>(max_size) / static_cast<double>(image.size()));
        if (scale < 1)
            dlib::resize_image(scale, image);
        else
            scale = 1.0;

        const double threshold = logit(confidence);
        const auto detections = net.process(image, threshold);
        py::list bboxes;
        for (const auto& d : detections)
        {
            py::dict bbox;
            bbox["xmin"] = d.rect.left() / scale;
            bbox["ymin"] = d.rect.top() / scale;
            bbox["xmax"] = d.rect.right() / scale;
            bbox["ymax"] = d.rect.bottom() / scale;
            bbox["label"] = d.label;
            bbox["confidence"] = sigmoid(d.detection_confidence);
            bboxes.append(std::move(bbox));
        }
        return bboxes;
    }

    private:
    detector::infer net;
};

PYBIND11_MODULE(wallyfinder, m)
{
    m.doc() = "A simple solver for \"Where's Wally\" images";
    m.attr("__version__") = "1.0.0";
    py::class_<WallyFinder>(m, "WallyFinder", "The Wally Finder Neural Network")
        .def(py::init<>(), "Construct the WallyFinder model")
        .def(
            "__call__",
            &WallyFinder::forward,
            py::arg("image"),
            py::arg("confidence") = 0.5,
            py::arg("max_size") = std::numeric_limits<size_t>::max(),
            "Get list of detections as dictionaries with:\n\
\"xmin\", \"ymin\", \"xmax\", \"ymax\": the coordinates of the bounding box.\n\
\"label\": the label of the detected object\
\"confidence\": the confidence of the detected object\n")
        .def(
            "clean",
            &WallyFinder::clean,
            "Clean the temporary state of the network and free unused memory");
}
