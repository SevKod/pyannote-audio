# MIT License
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional
import random
import torch
import torch.nn as nn
from torchaudio.transforms import Resample
from torch import Tensor
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.core.transforms_interface  import BaseWaveformTransform


class SpeedPerturbDiarization(BaseWaveformTransform):
    """
    Add random speed perturbation to the input (audio) and targets (annotations), 
    using a set of speed factors.

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000
    speed_factors : list, optional
        Defaults to [0.95,1.0,1.05]
        List of factors which will be selected at random for speed perturbation.

    """

    def __init__(
        self,
        sample_rate: Optional[int] = 16000,
        speed_factors: Optional[int] = [0.95,1.0,1.05]
    ):
        super().__init__(
            sample_rate=sample_rate,
        )

        # Initialize a list of resamplers
        resamplers_list = []
        for factor in speed_factors:
            config = {
                "orig_freq": sample_rate,
                "new_freq": sample_rate * factor,
            }
            resamplers_list.append(Resample(**config))
        
        # Create a list of tuples : [(speed_factor1, resampler1), (speed_factor2, resampler2),...]
        self.resamplers = list(zip(speed_factors,resamplers_list)) 

    def forward(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = 16000,
        targets: Tensor = None,
        target_rate: Optional[int] = None
    ) -> ObjectDict:
        
        #Consider single channel only
        samples = samples.squeeze(1) # [Batch, num_samples]
        targets = targets.squeeze(1) # [Batch, num_speakers, num_samples]

        batch_size, num_samples = samples.shape
        _ , num_window,num_spk = targets.shape 

        #Prepare the targets for speed perturbation
        idx_range = int(num_samples / num_window) #Needed to aggregate the targets

        #Aggregate the windows to retrieve original chunk size
        aggregated_targets = targets.unsqueeze(2).expand(batch_size, num_window, idx_range, num_spk) # [Batch, num_window, idx_range, num_spk]
        aggregated_targets = aggregated_targets.flatten(1, 2) # [Batch, num_samples, num_spk]
        aggregated_targets = aggregated_targets.transpose(1,2) # [Batch, num_spk, num_samples]
        target_num_sample = idx_range*num_window #Number of samples in the target after aggregation

        #Choose resamplers from the ones initialized, at random for each element of the batch
        resamplers_selected = random.choices(self.resamplers, k=batch_size)

        new_samples = torch.zeros(batch_size,num_samples)
        new_targets = torch.zeros(batch_size, num_spk, target_num_sample)

        for i,(speed_factor, resampler) in enumerate(resamplers_selected) :
            #Apply speed perturbation to the input audio samples
            resampled = resampler(samples[i,:])
            #Apply speed perturbation (interpolation) to the diarization targets
            interp_target = nn.functional.interpolate(aggregated_targets[i,:,:].unsqueeze(0),scale_factor=speed_factor,mode='nearest').squeeze(0)

            #Match sample sizes accordingly
            num_resampled = len(resampled)
            num_newtarget = interp_target.shape[-1]

            #For the input
            if num_resampled > num_samples : #Crop if longer than initially
                resampled = resampled[:num_samples]
            
            if num_resampled < num_samples : #Pad if shorter than initially
                pad_size = num_samples - num_resampled
                resampled = torch.nn.functional.pad(resampled, (0, pad_size))
            
            #For the targets
            if num_newtarget > target_num_sample :
                interp_target = interp_target[:,:target_num_sample]
            
            if num_newtarget < target_num_sample :
                pad_size = target_num_sample - num_newtarget
                interp_target = torch.nn.functional.pad(interp_target, (0, pad_size))
                
            new_samples[i,:] = resampled
            new_targets[i,:,:] = interp_target
            
        new_targets = new_targets.transpose(1,2)
        #Reconstruct the diarization target shape
        new_targets = new_targets.view(batch_size, num_window, idx_range, num_spk)
        new_targets = new_targets[:, :, 0, :] # [Batch, num_window, num_spk]

        return ObjectDict(
            samples=new_samples.unsqueeze(1),
            targets=new_targets,
        )
