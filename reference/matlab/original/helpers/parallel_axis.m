% calculate the parallel axis of 3 layer beam

function y_bar=parallel_axis(w,yp,ye,ya,hp,he,ha)
    
    y_bar=(ye.*w.*he.*(he./2)+ya.*w*ha.*(he+ha./2)+yp.*w.*hp.*(he+ha+hp./2))./(ye.*w.*he+ya.*w.*ha+yp.*w.*hp);
    
end