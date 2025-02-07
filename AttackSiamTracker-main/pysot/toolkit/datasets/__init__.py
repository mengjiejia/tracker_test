from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .uav import UAVDataset
from .visdrone1 import VISDRONED2018Dataset
from .dtb import DTB70Dataset
from .uavdt import UAVDTDataset
from .v4r import V4RDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAV123_10fps' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20l' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'VISDRONED2018' in name:
            dataset = VISDRONED2018Dataset(**kwargs)
        elif 'V4RFlight112' in name:
            dataset = V4RDataset(**kwargs)
        elif 'UAV123' in name:
            #dataset = UAVDataset(**kwargs)
            UAV10Dataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'DTB70' in name:
            dataset = DTB70Dataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

