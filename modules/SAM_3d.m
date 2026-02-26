function sam_map = SAM_3d(gt, rec)
    muxy = mean(gt.*rec, 3);
    muxx = mean(gt.*gt, 3);
    muyy = mean(rec.*rec, 3);
    
    sam_map = acosd(muxy./sqrt(muxx.*muyy));
end