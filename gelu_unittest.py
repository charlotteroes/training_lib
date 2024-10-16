import sys
from itertools import product

import torch
sys.path.append("../ppl.pmx/")
from torch_function import GELU

import kernel_library
from test_suit import TestSuit

def main():
    llm_module = kernel_library.module
    host_device = torch.device("cpu", 0)
    gpu_device = torch.device("cuda", 0)
    torch.set_default_device(gpu_device)

    # The parameter configuration to run the tests.
    approximate_type = [True, False]
    batch_sizes = [1, 10]
    features = [1, 4]
    planes = [[320, 240], [640, 480], [1280, 720], [1920, 1080]]

    # Number of test cases.
    numbers = len(approximate_type) * len(batch_sizes) * len(features) * len(planes)
    dist_threshold = 1e-2
    test_suit = TestSuit('ppl.llm.kernel.cuda', 'pmx', 'gelu', host_device,
                         gpu_device, numbers, dist_threshold)
    torch.manual_seed(1)
    for approximate, batch_size, feature, plane in product(
            approximate_type, batch_sizes, features, planes):
        test_suit.case_preprocessing(device="cuda", dtype="fp16", n=batch_size,
                                     c=feature, h=plane[0], w=plane[1],
                                     approximate=approximate)

        # Generation of input data based on random numbers.
        input = torch.randn([batch_size, feature, plane[0], plane[1]],
                            dtype=torch.half)
        gate = torch.randn([batch_size, feature, plane[0], plane[1]],
                           dtype=torch.half)

        # Run the operators both in ppl.pmx and ppl.llm.kernel.cuda.
        output0 = GELU.gelu(input, gate, approximate)
        output1 = llm_module.gelu(input, gate, approximate)

        # Verify the consistency of the output tensors based on a distance function.
        output0 = output0.to(host_device, dtype=torch.float32)
        output1 = output1.to(host_device, dtype=torch.float32)
        # print(output0)
        # print(output1)
        test_suit.case_postprocessing(True, 1e-2, 1e-2, input=input, gate=gate,
                                      pmx_output=output0, llm_output=output1)
    test_suit.summary()

if __name__ == "__main__":
    main()
