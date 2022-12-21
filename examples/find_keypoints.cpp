#include <iostream> 
#include <string>
#include <omp.h>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc < 2) {
        std::cerr << "Usage: ./find_keypoints input.jpg (or .png)\n";
        return 0;
    }

    int num_thread = -1;
    int max_threads = omp_get_max_threads();

    if (argc == 3) {
        num_thread = std::min(atoi(argv[2]), max_threads);
    }
    else {
        num_thread = max_threads;
    }

    omp_set_num_threads(num_thread);

    Image img(argv[1]);
    img =  img.channels == 1 ? img : rgb_to_grayscale(img);

    std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);
    Image result = sift::draw_keypoints(img, kps);
    result.save("result.jpg");

    std::cout << "Thread count: " << num_thread << "\n";
    std::cout << "Found " << kps.size() << " keypoints. Output image is saved as result.jpg\n";
    return 0;
}
