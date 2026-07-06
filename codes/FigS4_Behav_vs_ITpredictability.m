%% Exp.1
clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/FigS4_Behav_vs_ITpredictability'];
if ~exist(savedir,'dir');mkdir(savedir);end

imtype    = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
load([rootD '/rawdata/DNNlabels.mat'])
models(end-1:end) = {'BS_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k','BS_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'};
bs       = readmatrix([rootD '/rawdata/benchmark_scores.csv']);
bs_tmp   = readcell([rootD '/rawdata/benchmark_scores.csv']);
bs       = bs(:,2:end);
modelId  = bs_tmp(2:end,1);
bentchId = bs_tmp(1,2:end);
figW = 12.5;
figH = 7.8;
[~,batIdx] = ismember('IT',bentchId);
[~,batIdx_behav] = ismember('behavior_vision',bentchId);

score = nan(length(models),1);
score_behav = nan(length(models),1); 
for m = 1:length(models)
    [~,cnnIdx] = ismember(models{m}(4:end),modelId);
    if cnnIdx ~=0
        score(m,:) = bs(cnnIdx,batIdx);
        score_behav(m,:) = bs(cnnIdx,batIdx_behav);
    end
end

targvars = {'accuracy','consistency','both'};
for v = 1:length(targvars)
    targvar = targvars{v};
    figure
    for type = 1:length(imtype)
        load([rootD '/analysis/Fig3_large_scale_scatter/' imtype{type} '.mat'])
        accs_avg  = [accs' ;mean(hum_accs)]*100;
        kappa_avg = [mean(kappa,1)'; mean(hum_kappa)];
        subplot(2,3,type);hold on

        if strcmp(targvar,'accuracy')
            dist = abs(accs_avg(end)-accs_avg(1:end-1));
            ylim([0 100])
            y = 90;
            y_behav = 90;
            x_behav = 0.3;
            set(gca,'FontName','Arial');
            ylabel(['|\Delta' targvar '|'], 'Interpreter','tex', 'FontName','Arial');
            figlabel = 'Fig.S4a';
        elseif strcmp(targvar,'consistency')
            dist = kappa_avg(end) - kappa_avg(1:end-1);
            ylim([0 0.8])
            y = 0.7;
            y_behav = 0.7;
            x_behav = 0.3;
            ylabel(['\Delta' targvar], 'Interpreter','tex', 'FontName','Arial');
            figlabel = 'Fig.S4b';
        elseif strcmp(targvar,'both')
            [accsZ,accsM,accsS] = zscore(accs_avg(1:end-1));
            [kappZ,kappM,kappS] = zscore(kappa_avg(1:end-1));

            hum_accsZ = (accs_avg(end)-accsM)/accsS;
            hum_kappZ = (kappa_avg(end)-kappM)/kappS;
            accsD = hum_accsZ - accsZ;
            kappD = hum_kappZ - kappZ;

            dist = sqrt(accsD.^2 + kappD.^2);
            y = 9;
            x_behav = 0.3;
            y_behav = 9;
            ylim([0 10])
            ylabel('root-sum-square')
            figlabel = 'Fig.S4c';
        end

        [r,p] = corr(score(:,1),dist,'Rows','complete');
        [r_behav,p_behav] = corr(score_behav(:,1),dist,'Rows','complete');
        scatter(score(:,1),dist,10,score_behav(:,1),'o','filled','MarkerFaceAlpha',.8)
        caxis([0 0.55]) 
        hold on
        % scatter(score(topIdx,1),dist(topIdx),13,'o','filled','MarkerFaceColor','r')
        if p<0.05 && r<0
            text(.02,y,sprintf('r = %0.3f \np = %0.3f',r,p),'FontSize',8,'Color','m','FontWeight','bold')
        else
            text(.02,y,sprintf('r = %0.3f \np = %0.3f',r,p),'FontSize',8,'Color','m')
        end
        if p_behav<0.05 && r_behav<0
            text(x_behav,y_behav,sprintf('r = %0.3f \np = %0.3f',r_behav,p_behav),'FontSize',8,'Color','b','FontWeight','bold')
        else
            text(x_behav,y_behav,sprintf('r = %0.3f \np = %0.3f',r_behav,p_behav),'FontSize',8,'Color','b')
        end
        xlim([0 0.5])
        xticks(0:0.1:0.5)
        axis square
        ax = gca;
        ax.TickDir = 'out';
        ax.LineWidth = 1;
        ax.FontName = 'Helvetica';
        ax.FontSize = 6;
        ax.XTickLabelRotation = 0;
        xlabel('IT predictability')
    end
    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')
    exportgraphics(gcf, [savedir '/' targvar '_exp1.pdf'], 'ContentType', 'vector');
    saveas(gcf,[savedir '/' figlabel '_exp1.png'])
end

%% Exp.2

clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2025';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/FigS4_Behav_vs_ITpredictability'];
if ~exist(savedir,'dir');mkdir(savedir);end

imtype    =  {'Sparse' 'Dense'};
load([rootD '/rawdata/DNNlabels.mat'])
models(end-1:end) = {'BS_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k','BS_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'};
bs       = readmatrix([rootD '/rawdata/benchmark_scores.csv']);
bs_tmp   = readcell([rootD   '/rawdata/benchmark_scores.csv']);
bs       = bs(:,2:end);
modelId  = bs_tmp(2:end,1);
bentchId = bs_tmp(1,2:end);
figW = 4.17;
figH = 7.8;
[~,batIdx] = ismember('IT',bentchId);
[~,batIdx_behav] = ismember('behavior_vision',bentchId);

score = nan(length(models),1);
score_behav = nan(length(models),1); 
for m = 1:length(models)
    [~,cnnIdx] = ismember(models{m}(4:end),modelId);
    if cnnIdx ~=0
        score(m,:) = bs(cnnIdx,batIdx);
        score_behav(m,:) = bs(cnnIdx,batIdx_behav);
    end
end
[~,topIdx] = maxk(score_behav,5);

targvars = {'accuracy','consistency','both'};
for v = 1:length(targvars)
    targvar = targvars{v};
    figure
    for type = 1:length(imtype)
        load([rootD '/analysis/Fig4_large_scale_scatter/' imtype{type} '.mat'])
        accs_avg  = [accs' ;mean(hum_accs)]*100;
        kappa_avg = [mean(kappa,1)'; mean(hum_kappa)];
        subplot(2,1,type);hold on

        if strcmp(targvar,'accuracy')
            dist = abs(accs_avg(end)-accs_avg(1:end-1));
            ylim([0 100])
            y = 90;
            y_behav = 90;
            x_behav = 0.3;
            set(gca,'FontName','Arial');
            ylabel(['|\Delta' targvar '|'], 'Interpreter','tex', 'FontName','Arial');
            figlabel = 'Fig.S4a';
        elseif strcmp(targvar,'consistency')
            dist = kappa_avg(end) - kappa_avg(1:end-1);
            ylim([0 0.8])
            y = 0.7;
            y_behav = 0.7;
            x_behav = 0.3;
            ylabel(['\Delta' targvar], 'Interpreter','tex', 'FontName','Arial');
            figlabel = 'Fig.S4b';
        elseif strcmp(targvar,'both')
            [accsZ,accsM,accsS] = zscore(accs_avg(1:end-1));
            [kappZ,kappM,kappS] = zscore(kappa_avg(1:end-1));

            hum_accsZ = (accs_avg(end)-accsM)/accsS;
            hum_kappZ = (kappa_avg(end)-kappM)/kappS;
            accsD = hum_accsZ - accsZ;
            kappD = hum_kappZ - kappZ;

            dist = sqrt(accsD.^2 + kappD.^2);
            y = 22.5;
            x_behav = 0.3;
            y_behav = 22.5;
            ylim([0 25])
            ylabel('root-sum-square')
            figlabel = 'Fig.S4c';
        end

        [r,p] = corr(score(:,1),dist,'Rows','complete');
        [r_behav,p_behav] = corr(score_behav(:,1),dist,'Rows','complete');
        scatter(score(:,1),dist,10,score_behav(:,1),'o','filled','MarkerFaceAlpha',.8)
        hold on
        if p<0.05 && r<0
            text(.02,y,sprintf('r = %0.3f \np = %0.3f',r,p),'FontSize',8,'Color','m','FontWeight','bold')
        else
            text(.02,y,sprintf('r = %0.3f \np = %0.3f',r,p),'FontSize',8,'Color','m')
        end
        if p_behav<0.05 && r_behav<0
            text(x_behav,y_behav,sprintf('r = %0.3f \np = %0.3f',r_behav,p_behav),'FontSize',8,'Color','b','FontWeight','bold')
        else
            text(x_behav,y_behav,sprintf('r = %0.3f \np = %0.3f',r_behav,p_behav),'FontSize',8,'Color','b')
        end
        xlim([0 0.5])
        xticks(0:0.1:0.5)
        axis square
        ax = gca;
        ax.TickDir = 'out';
        ax.LineWidth = 1;
        ax.FontName = 'Helvetica';
        ax.FontSize = 6;
        ax.XTickLabelRotation = 0;
        xlabel('IT predictability')
    end

    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')
    exportgraphics(gcf, [savedir '/' targvar '_exp2.pdf'], 'ContentType', 'vector');
    saveas(gcf,[savedir '/' figlabel '_exp2.png'])
end
