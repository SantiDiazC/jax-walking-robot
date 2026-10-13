%% calculate Moment from piezo in 3 layer beam.(Code is newly written at 24.05.30)

function Mpiezo=MomentPiezo(w,d31,Volt,Yp,hp,he,ha,y_bar)
    
    Mpiezo=1/2.*(w.*Yp.*d31.*Volt./hp.*(2.*y_bar.*hp-2.*hp.*(hp+ha+he)+hp.^2));

end