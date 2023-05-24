## Base classes for dataset and datamodule 

For correct usage, the data should be structured in folder, where is train, val and optionaly test folder. 
For new dataset class it is neccessery to write new class, which inherit from BaseDataset. 

Following functions should be written. 


```python
class CustomDataset(BaseDataset):
    def __init__(self, data_path,
             drop_invalid_point_function=None,
             point_cloud_transform=None,
             n_points=None,
             apply_pillarization=True):
        super().__init__()

    def __len__(self):
        """
        For each dataset should be separetly written. 
        Returns:
            length of the dataset
        """
        raise NotImplementedError()

    def _get_point_cloud_pair(self, index):
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Args:
            index:

        Returns:
            t1_frame: pointcloud in shape [1, N, features]
            t0_frame: pointcloud in shape [1, N, features]
        """

        raise NotImplementedError()

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [1, 4, 4]
        """
        raise NotImplementedError()

    def _get_global_transform(self, index):
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to 
        global coordinates system.
        """
        return None

    def _get_flow(self, index):
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [1, N, channels].
        """
        return None
```

### Usega 

TODO


