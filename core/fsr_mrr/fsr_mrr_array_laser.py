import time
import math
import numpy as np
import torch

from gpib_control.santec_ecdl import SantecTSL510_OBand, SantecTSL510ECDL
from gpib_control.newport import Newport_2918c
from Labs_Scripts.ProbeStation.TestClasses.LaserSweep import LaserSweep
from gpib_control.K2600 import Keithley2600

# PowMet = Newport_2918c('A', simulate=False)
# PowMet.set_channel('A')
# time.sleep(1)
# PowMet.autorange = 0

# Santec1 = SantecTSL510_OBand(25)
# Santec1.timeout = 30000

# Santec2 = SantecTSL510_OBand(26)
# Santec2.timeout = 30000

# Rings = []
# Rings[0] = Keithley2600(22, False, 'B')
# Rings[0].set_source_voltage()
# Rings[0].set_current_compliance(0.020)

# Rings[1] = Keithley2600(22, False, 'A')
# Rings[1].set_source_voltage()
# Rings[1].set_current_compliance(0.020)

# Rings[2] = Keithley2600(10, False, 'B')
# Rings[2].set_source_voltage()
# Rings[2].set_current_compliance(0.020)

# Rings[3] = Keithley2600(10, False, 'A')
# Rings[3].set_source_voltage()
# Rings[3].set_current_compliance(0.020)

# DictData = LaserSweep(Santec, PowMet)
# wavelengths = [1290, 1300, 1310]
# powers = np.zeros_like(wavelengths)
# for wi, w in enumerate(wavelengths):
#     Santec.wavelength = w
#     time.sleep(1)
#     powers[wi] = PowMet.power
#     print(w, PowMet.power)

# for ring_idx, ring in enumerate(Rings):
#     ring.output_on()
#     ring.output_off()
#     ring.voltage = 1.0
#     print(f'power ring {ring_idx}, {ring.voltage * ring.current}')

class SingleFSR_MRR(object):
    def __init__(
        self,
        laser: SantecTSL510_OBand,
        source_meter: Keithley2600,
        # pow_met: Newport_2918c,
        wavelength,
    ):
        self.laser = laser
        self.laser.timeout = 30000

        self.source_meter = source_meter
        self.source_meter.set_source_voltage()
        self.source_meter.set_current_compliance(0.020)

        
        # if wavelength == 0:
        #     self.wavelength = self.get_wavelengths_by_laser_sweep()
        # else:
        self.wavelength = wavelength
        
        self.laser.wavelength = self.wavelength
        time.sleep(1)

    # def get_wavelengths_by_laser_sweep(self):
    #     DictData = LaserSweep(self.laser, self.pow_met)
    #     raise NotImplementedError
    #     wavelength = 0
    #     return wavelength
    
    def laser_set_power(self, power):
        assert power <= 1
        self.laser.power = power
        self.laser.open_shutter()
    
    def laser_close_shutter(self):
        # pass
        self.laser.close_shutter()

    def voltage_set(self, v):
        assert v <= 1.0
        self.source_meter.voltage = v
        self.source_meter.output_on()

    def voltage_off(self):
        self.source_meter.output_off()

### MRR_array
class FSR_MRR_Config_Laser:
    def __init__(self):
        ### Single FSR
        self.v_min = -1.0
        self.v_max = 1.0
        self.fsr_channels = 1
        self.mrr_in = 4
        self.mrr_out = 1
        self.pos_and_neg = True
        
        self.pow_met = Newport_2918c('A', simulate=False)
        self.pow_met.set_channel('A')
        time.sleep(1)
        self.pow_met.autorange = 0

        self.wavelength = [1289.64, 1290.52, 1291.84, 1292.84]
        mrr_rings = list()
        mrr_rings.append(
            SingleFSR_MRR(
                laser = SantecTSL510_OBand(25),
                source_meter = Keithley2600(22, False, 'B'),
                # pow_met=self.pow_met,
                wavelength=self.wavelength[0],
            )
        )
        mrr_rings.append(
            SingleFSR_MRR(
                laser=SantecTSL510_OBand(26),
                source_meter=Keithley2600(22, False, 'A'),
                # pow_met=self.pow_met,
                wavelength=self.wavelength[1],
            )
        )
        mrr_rings.append(
            SingleFSR_MRR(
                laser=SantecTSL510_OBand(27),
                source_meter=Keithley2600(10, False, 'B'),
                # pow_met=self.pow_met,
                wavelength=self.wavelength[2],
            )
        )
        mrr_rings.append(
            SingleFSR_MRR(
                laser=SantecTSL510_OBand(28),
                source_meter=Keithley2600(10, False, 'A'),
                # pow_met=self.pow_met,
                wavelength=self.wavelength[3],
            )
        )
        self.mrr_rings = mrr_rings

        self.input_loss = [0.5, 0.8, 0.6, 0.9] # in dBm

        self.init_test()
    
    def init_test(self):
        input = [0.5, 0.1, 0.1, 0.5]
        voltage = [0.2, 0.3, 0.4, 0.5]
        for idx, ring in enumerate(self.mrr_rings):
            ring.voltage_set(voltage[idx])
            if input[idx] > 0:
                input_power = 10 * math.log10(input[idx]) + self.input_loss[idx]
                ring.laser_set_power(input_power)
            elif input[idx] == 0:
                ring.laser_close_shutter()

            power = self.pow_met.power
            print(power)
            ring.laser_close_shutter()
            ring.voltage_off()
            print('wait')

    """
        only set pruned input
        input: torch.tensor, range[-1,1]
    """
    def all_pos_linear(self, input):
        ### set laser magnitude and wavelengths
        for idx, ring in enumerate(self.mrr_rings):
            if input[idx] > 0:
                input_power = 10 * math.log10(input[idx].item()) + self.input_loss[idx]
                ring.laser_set_power(input_power)
            elif input[idx] == 0:
                ring.laser_close_shutter()
            else:
                raise ValueError('input should be positive')
        ### readout output
        power = self.pow_met.power
        mW = 10 ** (power / 10)

        return mW
    
    """
        set pruned input and set pruned voltage
        input: torch.tensor, range[-1,1]
        voltage: torch.tensor, range[-1,1]
    """
    def mrr_array_linear(self, input, voltage):
        input_pos_inx = input >= 0
        input_neg_inx = input < 0
        voltage_pos_inx = voltage >= 0
        voltage_neg_inx = voltage < 0
        
        powers = torch.zeros(self.mrr_in)

        ### set rings positive voltage
        voltage_abs = voltage.abs()
        voltage_abs[voltage_neg_inx] = 0
        for idx, ring in enumerate(self.mrr_rings):
            ring.voltage_set(voltage_abs[idx].item())

        # W+ x+
        input_abs = input.abs()
        input_abs[input_neg_inx] = 0
        powers[0] = self.all_pos_linear(input_abs)
        
        # W+ x-
        input_abs = input.abs()
        input_abs[input_pos_inx] = 0
        powers[1] = self.all_pos_linear(input_abs)
        
        ### set rings negative voltage
        voltage_abs = voltage.abs()
        voltage_abs[voltage_pos_inx] = 0
        for idx, ring in enumerate(self.mrr_rings):
            ring.voltage_set(voltage_abs[idx].item())
        
        # W- x+
        input_abs = input.abs()
        input_abs[input_neg_inx] = 0
        
        powers[2] = self.all_pos_linear(input_abs)
        
        # W- x-
        input_abs = input.abs()
        input_abs[input_pos_inx] = 0
        powers[3] = self.all_pos_linear(input_abs)
        
        output = powers[0] - powers[1] - powers[2] + powers[3]
        
        return output

    """
        only set pruned input
        
        input: torch.tensor[4], range[-1,1]
        voltage: torch.tensor[4], range[-1,1]
    """
    # def mrr_array_linear(self, input, voltage):
    #     voltage_abs = voltage.abs()
        
    #     input_pos_inx = input >= 0
    #     input_neg_inx = input < 0
    #     voltage_pos_inx = voltage >= 0
    #     voltage_neg_inx = voltage < 0

    #     ### set rings voltage
        
    #     for idx, ring in enumerate(self.mrr_rings):
    #         ring.voltage_set(voltage_abs[idx].item())
        
    #     powers = torch.zeros(self.mrr_in)
        
    #     # W+ x+
    #     input_abs = input.abs()
    #     input_abs[voltage_neg_inx] = 0
    #     input_abs[input_neg_inx] = 0
    #     powers[0] = self.all_pos_linear(input_abs)
        
    #     # W+ x-
    #     input_abs = input.abs()
    #     input_abs[voltage_neg_inx] = 0
    #     input_abs[input_pos_inx] = 0
    #     powers[1] = self.all_pos_linear(input_abs)
        
    #     # W- x+
    #     input_abs = input.abs()
    #     input_abs[voltage_pos_inx] = 0
    #     input_abs[input_neg_inx] = 0
        
    #     powers[2] = self.all_pos_linear(input_abs)
        
    #     # W- x-
    #     input_abs = input.abs()
    #     input_abs[voltage_pos_inx] = 0
    #     input_abs[input_pos_inx] = 0
    #     powers[3] = self.all_pos_linear(input_abs)
        
    #     output = powers[0] - powers[1] - powers[2] + powers[3]
        
    #     return output
    
    """
        input: torch.tensor[4], range[-1,1], bz*din
        voltage: torch.tensor[4], range[-1,1], dout*din
    """
    def linear(self, input_mat, voltage_mat):
        input_mat.clamp(-1,1)
        voltage_mat.clamp(self.v_min, self.v_max)
        ### now only support 4*4
        dout, din = voltage_mat.size()
        bz = input_mat.size(0)
        
        assert(din == self.mrr_in)
        
        output_mat = torch.zeros(bz, dout)
        for out_idx in range(dout):
            voltage = voltage_mat[out_idx].squeeze()
            for bz_idx in range(bz):
                input = input_mat[bz_idx].squeeze()
                output_mat[bz_idx, out_idx] = self.mrr_array_linear(input, voltage)
        
        return output_mat

