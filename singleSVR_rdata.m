%%
clear
close all

% preprocessing_final;
load .\origdata_final\rData_V0
load .\origdata_final\rData_Cobre
load .\origdata_final\rData_Glu
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
clear rData_Cobre
clear rData_Glu
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
for repeat_test=1:10
    t1=clock;
    window=waitbar(0, ['0 times. This loop use: 0 minutes']);
    times=0;
    
    % Split dataset
    load (['.\origdata_final\44holdindx_',num2str(repeat_test)],'h');
    Train_PANSS = pos_all(training(h))';
    Test_PANSS = pos_all(test(h))';
    
    % Define trainset and testset
    rTrain_dat = rData_all(:,training(h))';
    rTest_dat = rData_all(:,test(h))';
    
    mse_mat=zeros(1,32);
    coef_mat=zeros(1,32);
    PANSS_mat=cell(1,32);
    
    % Grid Search
    zr=[0.003,0.005,0.007,0.01];
    i=0;
    
%     for r = 1:1:length(zr)
    r=3;
%----------------------- Set optional items -----------------------
        tic;
        i=i+1;
        k=0;
        rz=zr(r);

        opts=[];
        opts.init=2;
        % Termination 
        opts.tFlag=5;       % run .maxIter iterations
        opts.maxIter=100;   % maximum number of iterations
        % regularization
        opts.rFlag=1;       % use ratio
        opts.ind=[-1, -1, 1]';% leave nodes (each node contains one feature)
        %opts.G=[1:1:size(pData_train,1)];
        opts.nFlag=0;       % without normalization

        mse=zeros(1,286);
        coef=zeros(1,286);
        PANSS=cell(1,286);
        weight=cell(1,286);
        feaind=cell(1,286);
        category=cell(1,286);

        % Feature Selection
        [rx1, rfunVal1] = LeastR(rTrain_dat, Train_PANSS, rz, opts);
        rktt = find(abs(rx1)>0);
        %%
        % Create new feature matrix
        rP = rTrain_dat(:,rktt);
        rPt = rTest_dat(:,rktt);
       
        Y = Train_PANSS;
        Yt = Test_PANSS;
        
        % Calculate kernel coefficient
        sigma=1;
        rK=calckernel('linear',sigma,rP);
        rKT=calckernel('linear',sigma,rP,rPt);        
        k=k+1;
        K=rK;
        KT=rKT;
%         K=rP;
%         KT=rPt;
        
        % Training
        model= libsvmtrain(Y, [(1:length(Y))', K], '-s 4 -t 4'); % #ok<SVMTRAIN>

        [predscore,reg_mse,~] = libsvmpredict(Yt, [(1:length(Yt))', KT], model);
        PANSS{k}=predscore;
        mse(k)=reg_mse(2);
        coef(k)=reg_mse(3);

        %find the optimal coef in kernel weights-gird search
        [coef_max,ind]=max(coef);
        mse_min=mse(ind);
        Predict_PANSS =PANSS{ind};
        mse_mat(i)=mse_min;
        coef_mat(i)=coef_max;
        PANSS_mat{i}=Predict_PANSS;
        % clear coef ;clear mse;clear PANSS;
%         times=times+1;
%         timing=toc;
%         waitbar(times/(length(zr)), ...
%             window, [num2str(times),...
%             ' times. This loop use: ',num2str(timing/60),' minutes'])
%     end
    
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