clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir   = [rootD '/analysis/Fig5_ablation'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp2/summary.mat'])
cnntype   = {'resnet50','resnet50_abl','convnextL','convnextL_abl','cornet-s','cornet-s_abl','vit_l_16','vit_l_16_abl',...
    'resnet50-blur-st','resnet50-blur-st_abl','cornet-s-blur-st','cornet-s-blur-st_abl','resnet50-sin','resnet50-sin_abl',...
    'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384','convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384_abl', 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k_abl'};
cnnlabel = {'inResNet' 'inConvNeXt' 'inCORnet' 'inViT' 'blResNet' 'blCORnet' 'stResNet'  'ftConvNeXt' 'ftViT'};
Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
facec     = [Cres;Ccon;Ccor;Cvit;...
    Cres;Ccor;Cres;...
    Ccon;Cvit]/255;
 
mark = {'o' 'o' 'o' 'o'...
    '^' '^' 'v'...
    'square' 'square'};
imtype    =  {'Sparse' 'Dense'};
figlabel  = 'Fig.5d';
startcell = {'A' 'E'};
top   = 1;
figure
p = nan(1,2);
figW = 7.6;
figH = 8.1;
for type = 1:length(imtype)
    accs = nan(length(cnntype)/2,2);
    a1 = 1;
    a2 = 1;
    for cnn = 1:length(cnntype)
        [base, imName_all] = get_cnn_resp(rootD,strrep(cnntype{cnn},'_abl',''),'Sil',top);
        [dat, imName] = get_cnn_resp(rootD,cnntype{cnn},imtype{type},top);
        imName_all = cellfun(@(x) x(1:end-6),imName_all,'UniformOutput',false);
        imName     = cellfun(@(x) x(1:end-7),imName,'UniformOutput',false);
        exp2Idx    = ismember(imName_all,imName);
        dat = (dat(base(exp2Idx)==1));
        score = mean(dat)*100;
        if ~contains(cnntype{cnn},'_abl')
            accs(a1,1) = score;
            a1 = a1+1;
        else
            accs(a2,2) = score;
            a2 = a2+1;
        end
    end

    d_batch = squeeze(mean(d_indv_cl(:,type+1,:),1,'omitnan'))'*100;
    hum_accs_sd  = std(d_batch);
    subplot(1,2,type)
    hold on
    [~,p(type)] = ttest(accs(:,1),accs(:,2));
    yregion(mean(d_batch)-hum_accs_sd,mean(d_batch)+hum_accs_sd,'FaceColor',[.8 .8 .8])

    for c = 1:size(accs,1)
        scatter([1 2],accs(c,:),50,mark{c} ,'filled','MarkerFaceColor',facec(c,:),...
            'MarkerFaceAlpha',0.55,'MarkerEdgeColor',facec(c,:),'LineWidth',1)
        plot(accs(c,:),'-','Color',[facec(c,:) 0.5],'LineWidth',1,...
            'MarkerFaceColor',facec(c,:))
        xticklabels({'Control' 'Silenced'})
        xticks([1 2])
    end
    xlim([0.5 2.5])
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    ax.XTickLabelRotation = 0;
    plot([min(xlim) max(xlim)],[mean(d_batch) mean(d_batch)],'--','Color',[.5 .5 .5 .5],'LineWidth',2);
    plot([min(xlim) max(xlim)],[min(d_batch) min(d_batch)],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    plot([min(xlim) max(xlim)],[max(d_batch) max(d_batch)],':','Color',[.5 .5 .5 .5],'LineWidth',1);
    if  type == 1
        ylabel('Accuracy (%)')
    end

    t = cell(length(cnnlabel)+2,3);
    t(3:end,1) = cnnlabel;
    t{1,1} = imtype{type};
    t{2,2} = 'control_accuracy (%)';
    t{2,3} = 'silenced_accuracy (%)';
    t(3:end,2) = num2cell(accs(:,1));
    t(3:end,3) = num2cell(accs(:,2));
    writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel,...
        'WriteVariableNames',false,'Range',[startcell{type} num2str(1)]);
end
rectangle('Position',[1 1 figW figH], ...
    'EdgeColor','none','FaceColor','none');
set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
    'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
    'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')
 saveas(gcf,[savedir '/' figlabel '.png'])
exportgraphics(gcf, [savedir '/' figlabel '.pdf'], 'ContentType', 'vector');