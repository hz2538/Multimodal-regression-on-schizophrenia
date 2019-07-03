function preprocessing_final
load .\origdata_final\V0lharea
load .\origdata_final\V0rharea
load .\origdata_final\V0lhcurvind
load .\origdata_final\V0rhcurvind
load .\origdata_final\V0lhfoldind
load .\origdata_final\V0rhfoldind
load .\origdata_final\V0lhmeancurv
load .\origdata_final\V0rhmeancurv
load .\origdata_final\V0lhthickness
load .\origdata_final\V0rhthickness
load .\origdata_final\V0lhvolume
load .\origdata_final\V0rhvolume
load .\origdata_final\V0_FA
load .\origdata_final\V0_MD
load .\origdata_final\V0_FC
load .\origdata_final\V0_fALFF

load .\origdata_final\Cobrelharea
load .\origdata_final\Cobrerharea
load .\origdata_final\Cobrelhcurvind
load .\origdata_final\Cobrerhcurvind
load .\origdata_final\Cobrelhfoldind
load .\origdata_final\Cobrerhfoldind
load .\origdata_final\Cobrelhmeancurv
load .\origdata_final\Cobrerhmeancurv
load .\origdata_final\Cobrelhthickness
load .\origdata_final\Cobrerhthickness
load .\origdata_final\Cobrelhvolume
load .\origdata_final\Cobrerhvolume
load .\origdata_final\Cobre_FA
load .\origdata_final\Cobre_MD
load .\origdata_final\Cobre_FC
load .\origdata_final\Cobre_fALFF

load .\origdata_final\Glulharea
load .\origdata_final\Glurharea
load .\origdata_final\Glulhcurvind
load .\origdata_final\Glurhcurvind
load .\origdata_final\Glulhfoldind
load .\origdata_final\Glurhfoldind
load .\origdata_final\Glulhmeancurv
load .\origdata_final\Glurhmeancurv
load .\origdata_final\Glulhthickness
load .\origdata_final\Glurhthickness
load .\origdata_final\Glulhvolume
load .\origdata_final\Glurhvolume
load .\origdata_final\Glu_FA
load .\origdata_final\Glu_MD
load .\origdata_final\Glu_FC
load .\origdata_final\Glu_fALFF

V0_No = size(V0_FA,1);
Cobre_No = size(Cobre_FA,1);
Glu_No = size(Glu_FA,1);
group = [ones(1,V0_No), ones(1,Cobre_No), ones(1,Glu_No)];


Data_all_1_orig=[V0lharea;Cobrelharea;Glulharea]';
Data_all_2_orig=[V0rharea;Cobrerharea;Glurharea]';
Data_all_3_orig=[V0lhcurvind;Cobrelhcurvind;Glulhcurvind]';
Data_all_4_orig=[V0rhcurvind;Cobrerhcurvind;Glurhcurvind]';
Data_all_5_orig=[V0lhfoldind;Cobrelhfoldind;Glulhfoldind]';
Data_all_6_orig=[V0rhfoldind;Cobrerhfoldind;Glurhfoldind]';
Data_all_7_orig=[V0lhthickness;Cobrelhthickness;Glulhthickness]';
Data_all_8_orig=[V0rhthickness;Cobrerhthickness;Glurhthickness]';
Data_all_9_orig=[V0lhvolume;Cobrelhvolume;Glulhvolume]';
Data_all_10_orig=[V0rhvolume;Cobrerhvolume;Glurhvolume]';
Data_all_11_orig=[V0lhmeancurv;Cobrelhmeancurv;Glulhmeancurv]';
Data_all_12_orig=[V0rhmeancurv;Cobrerhmeancurv;Glurhmeancurv]';
Data_all_13_orig=[V0_FC;Cobre_FC;Glu_FC]';
Data_all_14_orig=[V0_FA;Cobre_FA;Glu_FA]';
Data_all_15_orig=[V0_MD;Cobre_MD;Glu_MD]';
Data_all_16_orig=[V0_fALFF;Cobre_fALFF;Glu_fALFF]';

% Data_all_1=[V0lharea;Cobrelharea;Glulharea]';
% Data_all_2=[V0rharea;Cobrerharea;Glurharea]';
% Data_all_3=[V0lhcurvind;Cobrelhcurvind;Glulhcurvind]';
% Data_all_4=[V0rhcurvind;Cobrerhcurvind;Glurhcurvind]';
% Data_all_5=[V0lhfoldind;Cobrelhfoldind;Glulhfoldind]';
% Data_all_6=[V0rhfoldind;Cobrerhfoldind;Glurhfoldind]';
% Data_all_7=[V0lhthickness;Cobrelhthickness;Glulhthickness]';
% Data_all_8=[V0rhthickness;Cobrerhthickness;Glurhthickness]';
% Data_all_9=[V0lhvolume;Cobrelhvolume;Glulhvolume]';
% Data_all_10=[V0rhvolume;Cobrerhvolume;Glurhvolume]';
% Data_all_11=[V0lhmeancurv;Cobrelhmeancurv;Glulhmeancurv]';
% Data_all_12=[V0rhmeancurv;Cobrerhmeancurv;Glurhmeancurv]';
% Data_all_13=[V0_FC;Cobre_FC;Glu_FC]';
% Data_all_14=[V0_FA;Cobre_FA;Glu_FA]';
% Data_all_15=[V0_MD;Cobre_MD;Glu_MD]';
% Data_all_16=[V0_fALFF;Cobre_fALFF;Glu_fALFF]';

%normalization
Data_all_1=my_norm(Data_all_1_orig);
Data_all_2=my_norm(Data_all_2_orig);
Data_all_3=my_norm(Data_all_3_orig);
Data_all_4=my_norm(Data_all_4_orig);
Data_all_5=my_norm(Data_all_5_orig);
Data_all_6=my_norm(Data_all_6_orig);
Data_all_7=my_norm(Data_all_7_orig);
Data_all_8=my_norm(Data_all_8_orig);
Data_all_9=my_norm(Data_all_9_orig);
Data_all_10=my_norm(Data_all_10_orig);
Data_all_11=my_norm(Data_all_11_orig);
Data_all_12=my_norm(Data_all_12_orig);
Data_all_13=my_norm(Data_all_13_orig);
Data_all_14=my_norm(Data_all_14_orig);
Data_all_15=my_norm(Data_all_15_orig);
Data_all_16=my_norm(Data_all_16_orig);

save .\origdata_final\Data_all_1 Data_all_1;
save .\origdata_final\Data_all_2 Data_all_2;
save .\origdata_final\Data_all_3 Data_all_3;
save .\origdata_final\Data_all_4 Data_all_4;
save .\origdata_final\Data_all_5 Data_all_5;
save .\origdata_final\Data_all_6 Data_all_6;
save .\origdata_final\Data_all_7 Data_all_7;
save .\origdata_final\Data_all_8 Data_all_8;
save .\origdata_final\Data_all_9 Data_all_9;
save .\origdata_final\Data_all_10 Data_all_10;
save .\origdata_final\Data_all_11 Data_all_11;
save .\origdata_final\Data_all_12 Data_all_12;
save .\origdata_final\Data_all_13 Data_all_13;
save .\origdata_final\Data_all_14 Data_all_14;
save .\origdata_final\Data_all_15 Data_all_15;
save .\origdata_final\Data_all_16 Data_all_16;
save .\origdata_final\group group;

Data_V0_1=Data_all_1(:,1:V0_No);
Data_V0_2=Data_all_2(:,1:V0_No);
Data_V0_3=Data_all_3(:,1:V0_No);
Data_V0_4=Data_all_4(:,1:V0_No);
Data_V0_5=Data_all_5(:,1:V0_No);
Data_V0_6=Data_all_6(:,1:V0_No);
Data_V0_7=Data_all_7(:,1:V0_No);
Data_V0_8=Data_all_8(:,1:V0_No);
Data_V0_9=Data_all_9(:,1:V0_No);
Data_V0_10=Data_all_10(:,1:V0_No);
Data_V0_11=Data_all_11(:,1:V0_No);
Data_V0_12=Data_all_12(:,1:V0_No);
Data_V0_13=Data_all_13(:,1:V0_No);
Data_V0_14=Data_all_14(:,1:V0_No);
Data_V0_15=Data_all_15(:,1:V0_No);
Data_V0_16=Data_all_16(:,1:V0_No);

Data_Cobre_1=Data_all_1(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_2=Data_all_2(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_3=Data_all_3(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_4=Data_all_4(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_5=Data_all_5(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_6=Data_all_6(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_7=Data_all_7(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_8=Data_all_8(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_9=Data_all_9(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_10=Data_all_10(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_11=Data_all_11(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_12=Data_all_12(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_13=Data_all_13(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_14=Data_all_14(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_15=Data_all_15(:,V0_No+1:V0_No+Cobre_No);
Data_Cobre_16=Data_all_16(:,V0_No+1:V0_No+Cobre_No);

Data_Glu_1=Data_all_1(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_2=Data_all_2(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_3=Data_all_3(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_4=Data_all_4(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_5=Data_all_5(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_6=Data_all_6(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_7=Data_all_7(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_8=Data_all_8(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_9=Data_all_9(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_10=Data_all_10(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_11=Data_all_11(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_12=Data_all_12(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_13=Data_all_13(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_14=Data_all_14(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_15=Data_all_15(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);
Data_Glu_16=Data_all_16(:,V0_No+Cobre_No+1:V0_No+Cobre_No+Glu_No);

rData_V0=[
          Data_V0_1;
          Data_V0_2;
          Data_V0_3;
          Data_V0_4;
          Data_V0_5;
          Data_V0_6;
          Data_V0_7;
          Data_V0_8;
          Data_V0_9;
          Data_V0_10;
          Data_V0_11;
          Data_V0_12];
cData_V0=[
          Data_V0_13];
xData_V0=[
          Data_V0_14;
          Data_V0_15];
pData_V0=[
          Data_V0_16];
      
rData_Cobre=[
          Data_Cobre_1;
          Data_Cobre_2;
          Data_Cobre_3;
          Data_Cobre_4;
          Data_Cobre_5;
          Data_Cobre_6;
          Data_Cobre_7;
          Data_Cobre_8;
          Data_Cobre_9;
          Data_Cobre_10;
          Data_Cobre_11;
          Data_Cobre_12];
cData_Cobre=[
          Data_Cobre_13];
xData_Cobre=[
          Data_Cobre_14;
          Data_Cobre_15];
pData_Cobre=[
          Data_Cobre_16];
      
rData_Glu=[
          Data_Glu_1;
          Data_Glu_2;
          Data_Glu_3;
          Data_Glu_4;
          Data_Glu_5;
          Data_Glu_6;
          Data_Glu_7;
          Data_Glu_8;
          Data_Glu_9;
          Data_Glu_10;
          Data_Glu_11;
          Data_Glu_12];
cData_Glu=[
          Data_Glu_13];
xData_Glu=[
          Data_Glu_14;
          Data_Glu_15];
pData_Glu=[
          Data_Glu_16];

      
save .\origdata_final\rData_V0 rData_V0
save .\origdata_final\cData_V0 cData_V0
save .\origdata_final\xData_V0 xData_V0
save .\origdata_final\pData_V0 pData_V0
save .\origdata_final\rData_Cobre rData_Cobre
save .\origdata_final\cData_Cobre cData_Cobre
save .\origdata_final\xData_Cobre xData_Cobre
save .\origdata_final\pData_Cobre pData_Cobre
save .\origdata_final\rData_Glu rData_Glu
save .\origdata_final\cData_Glu cData_Glu
save .\origdata_final\xData_Glu xData_Glu
save .\origdata_final\pData_Glu pData_Glu