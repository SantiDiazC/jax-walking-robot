%% calculate the 2nd moment inertia of 3 layer beam

function YI=inertia_2(w,yp,ye,ya,hp,he,ha,y_bar)
    
    YI=ye.*((w.*he.^3)/12+w.*he.*(he./2-y_bar).^2)+ya.*((w.*ha.^3)/12+w.*ha.*(he+ha./2-y_bar).^2)+yp.*((w.*hp^3)/12+w.*hp.*(he+ha+hp/2-y_bar).^2);

end