function x_clamped = clamp(x, minval, maxval)
    x_clamped = x;
    x_clamped(x_clamped < minval) = minval;
    x_clamped(x_clamped > maxval) = maxval;
end