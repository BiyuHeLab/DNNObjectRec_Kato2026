clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/Fig1_across_condition_error_consistency'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp1/summary.mat'])
load([rootD '/rawdata/DNNlabels.mat'])

imtype = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
nperm  = 10000;

% human
if ~exist([savedir '/error_consistency.mat'],'file')
    nbatch = size(d_batch_cl,3);
    ncnd = length(imtype);
    kappa_hum = nan(ncnd,ncnd,nbatch);
    pval  = kappa_hum;
    sigflag_hum = kappa_hum;
    for b = 1:nbatch
        for c1 = 1:ncnd
            for c2 = c1+1:ncnd
                dat1 = squeeze(d_batch_cl(:,c1+1,b));
                dat2 = squeeze(d_batch_cl(:,c2+1,b));
                excIdx = isnan(dat1)|isnan(dat2);
                dat1(excIdx) = [];
                dat2(excIdx) = [];
                dat1_av = mean(dat1);
                dat2_av = mean(dat2);
                c_exp = dat1_av*dat2_av + (1-dat1_av)*(1-dat2_av);
                c_obs = sum(dat1==dat2)/length(dat1);
                kappa_hum(c2,c1,b) = (c_obs - c_exp)/(1 - c_exp);

                kappa_perm = nan(nperm,1);
                for p = 1:nperm
                    dat1_perm = dat1(randsample(length(dat1),length(dat1)));
                    c_obs_perm = sum(dat1_perm==dat2)/length(dat1);
                    kappa_perm(p) = (c_obs_perm - c_exp)/(1 - c_exp);
                end
                pval(c2,c1,b) = (sum(kappa_perm>=kappa_hum(c2,c1))+1)/(nperm+1);
            end
        end
        [sigflag_hum(:,:,b), ~, ~, adj_p]=fdr_bh(pval(:,:,b));
    end
    cnntype = models;
    % DNN
    ncnd = length(imtype);
    sigflag_cnn = nan(ncnd,ncnd,length(cnntype));
    kappa_dnn = nan(ncnd,ncnd,length(cnntype));
    for cnn = 1:length(cnntype)
        ann_base = get_cnn_resp(rootD,cnntype{cnn},'Ori');

        pval  = nan(ncnd,ncnd);
        for c1 = 1:ncnd
            dat1_raw = get_cnn_resp(rootD,cnntype{cnn},imtype{c1});
            for c2 = c1+1:ncnd
                dat1 = dat1_raw;
                dat2 = get_cnn_resp(rootD,cnntype{cnn},imtype{c2});

                dat1 = dat1(ann_base == 1);
                dat2 = dat2(ann_base == 1);

                dat1_av = mean(dat1);
                dat2_av = mean(dat2);
                c_exp = dat1_av*dat2_av + (1-dat1_av)*(1-dat2_av);
                c_obs = sum(dat1==dat2)/length(dat1);
                kappa_dnn(c2,c1,cnn) = (c_obs - c_exp)/(1 - c_exp);

                kappa_perm = nan(nperm,1);
                for p = 1:nperm
                    dat1_perm = dat1(randsample(length(dat1),length(dat1)));
                    c_obs_perm = sum(dat1_perm==dat2)/length(dat1);
                    kappa_perm(p) = (c_obs_perm - c_exp)/(1 - c_exp);
                end
                pval(c2,c1) = (sum(kappa_perm>=kappa_dnn(c2,c1,cnn))+1)/(nperm+1);
            end
        end
        [sigflag_cnn(:,:,cnn), ~, ~, adj_p]=fdr_bh(pval);
    end
    save([savedir '/error_consistency.mat'],'sigflag_cnn','sigflag_hum','kappa_hum','kappa_dnn','imtype')
else
    load([savedir '/error_consistency.mat'],'sigflag_cnn','sigflag_hum','kappa_hum','kappa_dnn','imtype')
end
ncnd = length(imtype);
num_sig_cnn = sum(sigflag_cnn,3);
num_sig_hum = sum(sigflag_hum,3);
num_sig = cat(3,num_sig_hum,num_sig_cnn);


pval_grp = nan(ncnd,ncnd);
for c1 = 1:ncnd
    for c2 = c1+1:ncnd
        k = squeeze(kappa_hum(c2,c1,:));
        pval_grp(c2,c1) = signrank(k,0,"tail","right");
    end
end

p = pval_grp(~isnan(pval_grp));
[sigflag_tmp,~,~,p_fdr_tmp] = fdr_bh(p,0.05,'dep');

sigflag = nan(size(pval_grp));   
p_fdr   = nan(size(pval_grp));
mask = ~isnan(pval_grp);             
sigflag(mask) = sigflag_tmp; 
p_fdr(mask) = p_fdr_tmp;


for ii = 1:2
    if ii == 1
        t = 'Human';
        cbl = '#significant subjects';
        kappa_mean = mean(kappa_hum,3);
    elseif ii == 2
        t = 'DNN';
        cbl = '#significant DNNs';
        kappa_mean = mean(kappa_dnn,3);
    end
    figure
    RDM2plot = zeros(ncnd,ncnd) + tril(kappa_mean,-1); %tril(num_sig(:,:,ii),-1);
    RDM2plot(RDM2plot == 0) = NaN;
    h              = imagesc(RDM2plot);
    h.AlphaData    = isfinite(RDM2plot);
    h.Parent.Color = 'w';
    clim([0 0.4])
    xticklabels(imtype)
    xticks(1:length(imtype))
    yticklabels(imtype)
    yticks(1:length(imtype))
    % title(t)
    axis square
    box off
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 8;
    for c1=1:ncnd
        for c2 = c1+1:ncnd
            text(c1,c2,num2str(num_sig(c2,c1,ii)),'FontSize',8,'Color','r','HorizontalAlignment','center');
            idx = sub2ind([ncnd ncnd],c2,c1);
            if ii == 1 && sigflag(idx) == 1
                text(c1-0.25,c2-0.15,'*','FontSize',12,'Color','w','HorizontalAlignment','center');
            end
        end
    end
    figW = 4.75;
    figH = 4.75;
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])

    saveas(gcf,[savedir '/' t '.fig'])
    saveas(gcf,[savedir '/' t '.png'])
    exportgraphics(gcf, [savedir '/' t '.pdf'], 'ContentType', 'vector');
end

figure
ax = axes;
c = colorbar(ax);
ax.Visible = 'off';
c.Label.String = 'Error consistency';
clim([0 0.4])
figW = 3;
figH = 4;
set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
    'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
    'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
exportgraphics(gcf, [savedir '/error_consistency_CB.pdf'], 'ContentType', 'vector');


% Source data
hum_mean = mean(kappa_hum,3);
dnn_mean = mean(kappa_dnn,3);
t = cell(nchoosek(length(imtype),2),3);
t{1,2} = "human_mean";
t{1,3} = 'DNN_mean';

tt = 2;
for t1 = 1:length(imtype)-1
    for t2 = t1+1:length(imtype)
        t{tt,1} = [imtype{t1} '-' imtype{t2}];
        t{tt,2} = hum_mean(t2,t1);
        t{tt,3} = dnn_mean(t2,t1);
        tt = tt + 1;
    end
end

writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet','Fig.1e','WriteVariableNames',false);


