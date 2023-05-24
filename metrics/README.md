### Motion Flow Metrics

- AEE average endpoint error (EE) across all points
- AccS point ratio where EE < 0.05 or relative error < 0.05 
- AccR point ratio where EE < 0.1 or relative error < 0.1 
- Outl point ratio where EE > 0.3 or relative error > 0.1 
- ROutl point ratio where EE > 0.3 and relative error > 0.3
- AEE 50-50 as the average between the AEE measured separately on stationary and on moving points. Ground truth
odometry is used to compute the non-rigid flow component
$\bold{f}_{nr,gt,i} = \bold{f}_{gt,i} − (O_1 − I_4)pi$. Points with a non-rigid t→t+1
flow larger than mthresh = 5cm (corresponds to 1.8 $\frac{km}{h}$ ) are labeled as dynamic, the rest as stationary.

### Segmentation metrics 
- IoU (intersection over union)