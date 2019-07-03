clear
close all

% preprocessing_final;
load .\origdata_final\rData_V0
load .\origdata_final\cData_V0
load .\origdata_final\xData_V0
load .\origdata_final\pData_V0
load .\origdata_final\rData_Cobre
load .\origdata_final\cData_Cobre
load .\origdata_final\xData_Cobre
load .\origdata_final\pData_Cobre
load .\origdata_final\rData_Glu
load .\origdata_final\cData_Glu
load .\origdata_final\xData_Glu
load .\origdata_final\pData_Glu
load .\origdata_final\group;

load .\origdata_final\PANSS_pos_V0
load .\origdata_final\PANSS_neg_V0
load .\origdata_final\PANSS_gen_V0
load .\origdata_final\PANSS_tot_V0
load .\origdata_final\PANSS_pos_Cobre
load .\origdata_final\PANSS_neg_Cobre
load .\origdata_final\PANSS_gen_Cobre
load .\origdata_final\PANSS_tot_Cobre
load .\origdata_final\PANSS_pos_Glu
load .\origdata_final\PANSS_neg_Glu
load .\origdata_final\PANSS_gen_Glu
load .\origdata_final\PANSS_tot_Glu

rData_all=[rData_V0,rData_Cobre,rData_Glu]; % Structural MRI - Geometry 2*74*6
cData_all=[cData_V0,cData_Cobre,cData_Glu]; % fMRI - Functional Connectivity 116*115/2
xData_all=[xData_V0,xData_Cobre,xData_Glu]; % DTI - Fractional Anistropy & Mean Diffusivity 50+50
pData_all=[pData_V0,pData_Cobre,pData_Glu]; % fMRI - Voxelwise fALFF (mask) 61*73*61-OuterBrain -> 70831 

pos_all=[PANSS_pos_V0;PANSS_pos_Cobre;PANSS_pos_Glu]'; 
neg_all=[PANSS_neg_V0;PANSS_neg_Cobre;PANSS_neg_Glu]'; 
gen_all=[PANSS_gen_V0;PANSS_gen_Cobre;PANSS_gen_Glu]'; 
tot_all=[PANSS_tot_V0;PANSS_tot_Cobre;PANSS_tot_Glu]'; 

mse_repeat=zeros(1,1);
coef_repeat=zeros(1,1);
Predict_PANSS_repeat=cell(1,1);
weights_repeat=cell(1,1);
feainds_repeat=cell(1,1);
categories_repeat=cell(1,1);

clear rData_V0
clear cData_V0
clear xData_V0
clear pData_V0
clear rData_Cobre
clear cData_Cobre
clear xData_Cobre
clear pData_Cobre
clear rData_Glu
clear cData_Glu
clear xData_Glu
clear pData_Glu
clear PANSS_pos_V0
clear PANSS_neg_V0
clear PANSS_gen_V0
clear PANSS_tot_V0
clear PANSS_pos_Cobre
clear PANSS_neg_Cobre
clear PANSS_gen_Cobre
clear PANSS_tot_Cobre
clear PANSS_pos_Glu
clear PANSS_neg_Glu
clear PANSS_gen_Glu
clear PANSS_tot_Glu

%%
save('rData.mat','rData_all')
save('cData.mat','cData_all')
save('xData.mat','xData_all')
save('pData.mat','pData_all')
save('posPANSS.mat','pos_all')
save('negPANSS.mat','neg_all')
save('genPANSS.mat','gen_all')
save('totPANSS.mat','tot_all')

%% Training
for repeat_test=1:10
    %% Create waitbar
    repeat_test = 1;
    t1=clock;
    window=waitbar(0, ['0 times. This loop use: 0 minutes']);
    times=0;
    
    %% Split dataset
    %h= cvpartition(group,'h',44);
    %save (['.\origdata_final\44holdindx_',num2str(repeat_test)],'h');
    load (['.\origdata_final\44holdindx_',num2str(repeat_test)],'h');
    
    % Define which PANSS score as label
    Train_PANSS = pos_all(training(h))';
    Test_PANSS = pos_all(test(h))';
    
    % Define trainset and testset
    rTrain_dat = rData_all(:,training(h))';
    rTest_dat = rData_all(:,test(h))';
    cTrain_dat = cData_all(:,training(h))';
    cTest_dat = cData_all(:,test(h))';
    pTrain_dat = pData_all(:,training(h))';
    pTest_dat = pData_all(:,test(h))';
    xTrain_dat = xData_all(:,training(h))';
    xTest_dat = xData_all(:,test(h))';
    
    mse_mat=zeros(1,32);
    coef_mat=zeros(1,32);
    PANSS_mat=cell(1,32);
    
    %% Grid Search
    zr=[0.2,0.3,0.4]; % 0.1-0.7
    zc=[0.6,0.7,0.8]; % 0.1-0.4 0.7-0.9
    zp=[0.5,0.6,0.7]; % 0.2-0.3 0.6-0.9
    zx=[0.7,0.8,0.9]; % 0.1-0.3 0.5 0.6 0.8 0.9
    i=0;
    
%     [R, C, P, X] = ndgrid(zr, zc, zp, zx);
%     [a, b] = arrayfun(@(R) LeastR(rTrain_dat, Train_PANSS, R), R);
    
    %%
for r = 1:1:length(zr)
    for c = 1:1:length(zc)
        for p = 1:1:length(zp)
            for x= 1:1:length(zx)
%----------------------- Set optional items -----------------------
        %%
        r=1, c=1, p=1, x=1;
        tic;
        i=i+1;
        k=0;
        rz=zr(r);
        cz=zc(c);
        pz=zp(p);
        xz=zx(x);
        opts=[];

        % Starting point
        opts.init=2;

        % Termination 
        opts.tFlag=5;       % run .maxIter iterations
        opts.maxIter=100;   % maximum number of iterations

        % regularization
        opts.rFlag=1;       % use ratio

        opts.ind=[-1, -1, 1]';% leave nodes (each node contains one feature)
        %opts.G=[1:1:size(pData_train,1)];

        % Normalization
        opts.nFlag=0;       % without normalization

        mse=zeros(1,286);
        coef=zeros(1,286);
        PANSS=cell(1,286);
        weight=cell(1,286);
        feaind=cell(1,286);
        category=cell(1,286);

        %%
        [rx1, rfunVal1]= LeastR(rTrain_dat, Train_PANSS, rz, opts);
        [cx1, cfunVal1]= LeastR(cTrain_dat, Train_PANSS, cz, opts);
        [px1, pfunVal1]= LeastR(pTrain_dat, Train_PANSS, pz, opts);
        [xx1, xfunVal1]= LeastR(xTrain_dat, Train_PANSS, xz, opts);
        rktt = find(abs(rx1)>0);
        cktt = find(abs(cx1)>0);
        pktt = find(abs(px1)>0);
        xktt = find(abs(xx1)>0);
        
        %%
        rP = rTrain_dat(:,rktt);
        rPt = rTest_dat(:,rktt);
        cP = cTrain_dat(:,cktt);
        cPt = cTest_dat(:,cktt);
        pP = pTrain_dat(:,pktt);
        pPt = pTest_dat(:,pktt);
        xP = xTrain_dat(:,xktt);
        xPt = xTest_dat(:,xktt);

%         %% Normalization
%         % column normalization
%         rP=(rP-repmat(mean(rP),[size(rP,1) 1]))./repmat(std(rP),[size(rP,1) 1]);                
%         rPt=(rPt-repmat(mean(rPt),[size(rPt,1) 1]))./repmat(std(rPt),[size(rPt,1) 1]);  
% 
%         % row normalization
%         rP=rP./repmat(sqrt(sum(rP.^2,2)),[1 size(rP,2)]);
%         rPt= rPt./repmat(sqrt(sum(rPt.^2,2)),[1 size(rPt,2)]); 
%         
%         % column normalization        
%         cP=(cP-repmat(mean(cP),[size(cP,1) 1]))./repmat(std(cP),[size(cP,1) 1]);     
%         cPt=(cPt-repmat(mean(cPt),[size(cPt,1) 1]))./repmat(std(cPt),[size(cPt,1) 1]); 
% 
%         % row normalization
%         cP=cP./repmat(sqrt(sum(cP.^2,2)),[1 size(cP,2)]);
%         cPt= cPt./repmat(sqrt(sum(cPt.^2,2)),[1 size(cPt,2)]);
%         
%         % column normalization           
%         pP=(pP-repmat(mean(pP),[size(pP,1) 1]))./repmat(std(pP),[size(pP,1) 1]);                
%         pPt=(pPt-repmat(mean(pPt),[size(pPt,1) 1]))./repmat(std(pPt),[size(pPt,1) 1]);    
% 
%         % row normalization
%         pP=pP./repmat(sqrt(sum(pP.^2,2)),[1 size(pP,2)]);
%         pPt= pPt./repmat(sqrt(sum(pPt.^2,2)),[1 size(pPt,2)]);  
%         
%         % column normalization          
%         xP=(xP-repmat(mean(xP),[size(xP,1) 1]))./repmat(std(xP),[size(xP,1) 1]);                
%         xPt=(xPt-repmat(mean(xPt),[size(xPt,1) 1]))./repmat(std(xPt),[size(xPt,1) 1]);  
% 
%         % row normalization
%         xP=xP./repmat(sqrt(sum(xP.^2,2)),[1 size(xP,2)]);
%         xPt= xPt./repmat(sqrt(sum(xPt.^2,2)),[1 size(xPt,2)]); 
%         
%         Y = Train_PANSS;
%         Yt = Test_PANSS;
%         
%         sigma=1; 
%         rK=calckernel('linear',sigma,rP);
%         rKT=calckernel('linear',sigma,rP,rPt);
%         cK=calckernel('linear',sigma,cP);
%         cKT=calckernel('linear',sigma,cP,cPt); 
%         pK=calckernel('linear',sigma,pP);
%         pKT=calckernel('linear',sigma,pP,pPt);
%         xK=calckernel('linear',sigma,xP);
%         xKT=calckernel('linear',sigma,xP,xPt);
        
        %%
    for clu=0:10, c4=clu*0.1;
        for row=0:10-clu,             %row=0:10-clu;
            for col=0:10-clu-row,         %col=0:10-clu-row;
                 k=k+1;
                 c1=row*0.1;
                 c2=col*0.1;
                 c3=1-c2-c1-c4;
                 K=c1*rK+c2*cK+c3*pK+c4*xK;
                 KT=c1*rKT+c2*cKT+c3*pKT+c4*xKT;
                 model= svmtrain(Y, [(1:length(Y))', K], '-s 4 -t 4');
                 alpha=zeros(length(Y),1);
                 alpha(model.SVs)=model.sv_coef;
                 ft=KT*alpha;
                 %bt=breakeven(ft,Yt,@pre_rec_equal);
                 %et(fd,1)=evaluate(sign(ft-bt),Yt);
                 [predscore,reg_mse,~] = svmpredict(Yt, [(1:length(Yt))', KT], model);
                 PANSS{k}=predscore;
                 mse(k)=reg_mse(2); 
                 coef(k)=reg_mse(3); 
            end
        end
    end
%find the optimal coef in kernel weights-gird search
[coef_max,ind]=max(coef);
mse_min=mse(ind);
Predict_PANSS =PANSS{ind};
mse_mat(i)=mse_min;
coef_mat(i)=coef_max;
PANSS_mat{i}=Predict_PANSS;
% clear coef ;clear mse;clear PANSS;
times=times+1;
timing=toc;
waitbar(times/(length(zr)*length(zc)*length(zp)*length(zx)), ...
    window, [num2str(times),...
    ' times. This loop use: ',num2str(timing/60),' minutes'])
            end
        end
    end
end    
close(window)
%find the optimal coef in z-grid search
[coef_mat_max,matind]=max(coef_mat);
mse_mat_min=mse_mat(matind);
Predict_PANSS_mat=PANSS_mat{matind};
% clear coef_mat ;clear mse_mat;clear PANSS_mat;
mse_repeat(repeat_test)=mse_mat_min;
coef_repeat(repeat_test)=coef_mat_max;
Predict_PANSS_repeat{repeat_test}=Predict_PANSS_mat;
ztest(repeat_test)=matind;
%--------------------------------------------------------------------------
% Create a scatter Diagram
disp('Create a scatter Diagram')
figure,
% plot the 1:1 line
plot(Yt,Yt,'LineWidth',3);

hold on
scatter(Yt,Predict_PANSS_mat,'filled');
hold off
grid on

set(gca,'FontSize',18)
xlabel('Actual','FontSize',18)
ylabel('Estimated','FontSize',18)
title(['The No.',num2str(repeat_test),...
    ' test. R^2=' num2str(coef_mat_max^2,4)],...
    'FontSize',25)

drawnow

fn='ScatterDiagram';
fnpng=[fn,'.png'];
% print('-dpng',fnpng);

disp(['complete the No.',num2str(repeat_test), ...
    ' test! The running time for this test is : ',...
    num2str(etime(clock,t1)/3600), ' hours']);
    
end

%the final stats
mse_avr=mean(mse_repeat);
mse_std=std(mse_repeat);
coef_avr=mean(coef_repeat);
coef_std=std(coef_repeat);
r2_repeat=coef_repeat.^2;
r2_avr=mean(r2_repeat);
r2_std=std(r2_repeat);
rmse_repeat=sqrt(mse_repeat);
rmse_avr=mean(rmse_repeat);
rmse_std=std(rmse_repeat);