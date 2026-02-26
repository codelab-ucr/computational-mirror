function spectra = RL_invert(meas, A, iters)
    latent_est = A'*meas;
    
    for idx = 1:iters
        est = A*latent_est;
        rel_blur = meas./est;
        err = A'*rel_blur;
        latent_est = latent_est.*err;
    end
    spectra = latent_est;
end