from compressai.dna_entropy_coding.huffman_coder  import HuffmanCoder
from compressai.dna_entropy_coding.goldman_coder  import GoldmanCoderDNA
import numpy as np
import torch
import math

NON_TRIVIAL_CHANNEL_QUALITY_1 = [24, 29, 35, 51, 55, 58, 74, 79, 81, 86, 105, 106, 109, 111, 128, 129, 136, 146, 151, 160, 164, 175, 176, 178, 191]
NON_TRIVIAL_CHANNEL_QUALITY_2 = [7, 26, 29, 35, 37, 39, 51, 53, 55, 58, 74, 79, 81, 86, 93, 101, 105, 106, 109, 111, 122, 127, 128, 129, 136, 146, 151, 158, 160, 164, 166, 175, 178, 185, 189, 191]
NON_TRIVIAL_CHANNEL_QUALITY_3 = [0, 3, 7, 11, 16, 17, 21, 29, 35, 37, 39, 50, 51, 53, 54, 55, 58, 66, 68, 74, 79, 81, 83, 86, 93, 101, 105, 106, 109, 111, 118, 122, 124, 127, 128, 129, 136, 145, 146, 151, 158, 160, 164, 166, 167, 173, 175, 178, 185, 186, 189, 190, 191]
NON_TRIVIAL_CHANNEL_QUALITY_4 = [0, 1, 3, 5, 7, 10, 11, 16, 17, 21, 24, 25, 26, 27, 29, 30, 35, 37, 39, 40, 42, 49, 50, 51, 53, 54, 55, 58, 59, 66, 67, 71, 73, 74, 75, 76, 79, 81, 82, 83, 85, 86, 90, 93, 98, 100, 101, 105, 106, 109, 110, 111, 115, 118, 122, 124, 127, 128, 129, 136, 145, 146, 149, 151, 157, 158, 160, 164, 166, 167, 173, 175, 178, 185, 186, 188, 189, 191]
NON_TRIVIAL_CHANNEL_QUALITY_5 = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 15, 16, 17, 21, 24, 25, 26, 27, 29, 30, 31, 35, 37, 38, 39, 40, 42, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 66, 67, 71, 73, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 90, 92, 93, 98, 99, 100, 101, 103, 104, 105, 106, 109, 110, 111, 112, 113, 115, 118, 122, 123, 124, 126, 127, 128, 129, 136, 139, 140, 143, 145, 146, 147, 149, 150, 151, 157, 158, 160, 162, 164, 166, 167, 168, 170, 173, 174, 175, 177, 178, 180, 185, 186, 188, 189, 191]
NON_TRIVIAL_CHANNEL_QUALITY_6 = [1, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 33, 34, 41, 42, 45, 46, 48, 49, 53, 55, 57, 60, 62, 63, 66, 67, 70, 72, 74, 76, 79, 80, 85, 86, 92, 93, 100, 103, 104, 105, 107, 108, 109, 111, 112, 117, 118, 119, 122, 123, 124, 129, 130, 132, 134, 136, 137, 139, 140, 143, 144, 150, 151, 152, 156, 158, 159, 160, 162, 164, 167, 168, 169, 170, 171, 172, 178, 181, 184, 186, 187, 188, 189, 190, 191, 193, 194, 196, 197, 198, 200, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 218, 219, 220, 224, 227, 228, 229, 231, 232, 233, 234, 235, 237, 238, 240, 242, 243, 247, 248, 252, 253, 256, 260, 261, 262, 266, 267, 268, 270, 274, 277, 278, 279, 280, 283, 285, 286, 288, 289, 290, 293, 294, 295, 296, 297, 298, 299, 301, 302, 303, 307, 309, 310, 311, 313, 314]
NON_TRIVIAL_CHANNEL_QUALITY_7 = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 38, 41, 42, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 66, 67, 70, 72, 74, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 89, 92, 93, 95, 100, 103, 104, 105, 106, 107, 108, 109, 111, 117, 118, 119, 121, 122, 123, 125, 126, 128, 129, 130, 132, 134, 136, 137, 139, 140, 143, 144, 146, 150, 151, 152, 156, 158, 159, 160, 162, 164, 166, 167, 168, 169, 170, 171, 174, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 217, 218, 219, 220, 224, 225, 226, 227, 228, 229, 231, 233, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 247, 248, 250, 252, 253, 254, 256, 258, 260, 261, 262, 263, 266, 267, 268, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 283, 285, 287, 288, 289, 291, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 306, 307, 308, 309, 310, 311, 312, 313, 315, 316]
NON_TRIVIAL_CHANNEL_QUALITY_8 = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 92, 93, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 115, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 149, 150, 151, 152, 153, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 224, 225, 226, 227, 228, 229, 231, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 247, 248, 250, 252, 253, 254, 256, 257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 319]
NON_TRIVIAL_CHANNELS = [NON_TRIVIAL_CHANNEL_QUALITY_1, NON_TRIVIAL_CHANNEL_QUALITY_2, NON_TRIVIAL_CHANNEL_QUALITY_3,
                       NON_TRIVIAL_CHANNEL_QUALITY_4, NON_TRIVIAL_CHANNEL_QUALITY_5, NON_TRIVIAL_CHANNEL_QUALITY_6,
                       NON_TRIVIAL_CHANNEL_QUALITY_7, NON_TRIVIAL_CHANNEL_QUALITY_8]

class Coder:
    """Coder to encode and decode a rounded tensor at a given quality level 
    into DNA whil respecting the constraint of homopolymers. It contains a Huffman
    coder and a Goldman coder, according to the algorithm.
    """

    
    def __init__(self): 
        self.gold_coder = GoldmanCoderDNA()
        self.huffman_coder = None


    def encode(self, y_hat_quantized, quality_level):
        """Encodes the non trivial channels of the quantized integer tensor into DNA.

        :y_hat_quantized: the quantized (rounded) latent space representation of the image x
        :quality_level: integer between 1 and 8, defines the channels to encode
        """

        rounded_without_trivial_channels = torch.Tensor()
        for i in NON_TRIVIAL_CHANNELS[quality_level-1]: #only encode non trivial channels
            rounded_without_trivial_channels = torch.cat((rounded_without_trivial_channels, torch.unsqueeze(y_hat_quantized[0][i], dim=0)))

        rounded_without_trivial_channels = rounded_without_trivial_channels.int()
            
        elements, frequencies = torch.unique(rounded_without_trivial_channels, return_counts=True)
        self.huffman_coder = HuffmanCoder(elements.tolist(), frequencies.tolist(), 3, verbose=0)
        to_encode = torch.clone(rounded_without_trivial_channels)

        flattened = to_encode.flatten().tolist()
        flattened = [str(v) for v in flattened] #huffman_coder encodes strings, not integers

        ternary_huffman_code = self.huffman_coder.encode(flattened)
        dna = self.gold_coder.encode(ternary_huffman_code)
        return dna

    def decode(self, dna_seq, quality_level, img_shape):
        """Decodes the DNA string and puts in back in the shape of a quantized latent 
        space integer tensor.

        :dna_seq: the encoded DNA string
        :quality_level: integer between 1 and 8, defines the non trivial channels in order to
                        reconstruct the latent space vector
        :img_shape: shape of the original image
        """

        #the autoencoder yields a tensor of size 1 x n x height/16 x width/16, with padding if the shape was not divisible by 16
        shape = (math.ceil(img_shape[2]/16), math.ceil(img_shape[3]/16))

        non_trivial_channels = NON_TRIVIAL_CHANNELS[quality_level-1]
        number_of_channels = 192 if quality_level <= 5 else 320

        ternary_decoded = self.gold_coder.decode(dna_seq)
        decoded = self.huffman_coder.decode(ternary_decoded) 

        float_decoded = [float(v) for v in decoded]
        float_decoded = torch.Tensor((float_decoded)).reshape((len(non_trivial_channels), shape[0], shape[1]))
        full_decoded = torch.Tensor()
        
        j = 0    #counter of non trivial channel to add
        for i in range(number_of_channels): #reconstruct by adding null channels
            if i in non_trivial_channels:
                full_decoded = torch.cat((full_decoded, torch.unsqueeze(float_decoded[j], dim=0)))
                j += 1
            else:
                full_decoded = torch.cat((full_decoded, torch.zeros((1, shape[0], shape[1]))))

        return torch.unsqueeze(full_decoded, dim=0)

