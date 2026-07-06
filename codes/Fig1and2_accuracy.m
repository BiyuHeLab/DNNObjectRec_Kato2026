clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir = [rootD '/analysis/Fig1_Fig2_accuracy'];
if ~exist(savedir,'dir');mkdir(savedir);end
load([rootD '/rawdata/behav_exp1/summary.mat'])
cnntype   = {'resnet50','convnextL','cornet-s','vit_l_16',...
    '','resnet50-blur-st','cornet-s-blur-st','resnet50-sin',...
    '','clip_convnextL_image','clip_vit-l-laion_image', ....
    '','convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384' 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k',''};
cnnLabel  = {'inResNet','inConvNeXt','inCORnet','inViT',...
    '','blResNet','blCORnet','stResNet',...
    '','zsConvNeXt','zsViT'...
    '','ftConvNeXt' 'ftViT'  ,'','Human'};

Cres = [49 130 189];
Ccon = [59 10 117];
Ccor = [0 178 0];
Cvit = [255 105 180];
Chum = [100 100 100];

facec     = [Cres;Ccon;Ccor;Cvit;...
             Cres;Ccor;Cres;...
             Ccon;Cvit;...
             Ccon;Cvit];
imtype   = {'Ori' 'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
figlabel = {'Fig.1d' 'Fig.2a' 'Fig.2b' 'Fig.2c' 'Fig.2d' 'Fig.2e' 'Fig.2f'};
top   = 1;

figW = 6.5;
figH = 4.875;
for type = 1:length(imtype) 
    for cnn = 1:length(cnntype)
        if  strcmp(cnntype{cnn},'')
            score = NaN;
        else
            base = get_cnn_resp(rootD,cnntype{cnn},'Ori',top);
            dat  = get_cnn_resp(rootD,cnntype{cnn},imtype{type},top);
            if ~strcmp(imtype{type},'Ori')
                dat = (dat(base==1));
            end
            score = mean(dat)*100;
        end
        if cnn == 1
            accs = score;
        else
            accs = cat(2,accs,score);
        end
    end

    figure
    if strcmp(imtype{type},'Ori')
        ast_y = 100;
        d_batch = squeeze(mean(d_indv(:,type,:),1,'omitnan'))'*100;
    else
        d_batch = squeeze(mean(d_batch_cl(:,type,:),1,'omitnan'))'*100;
        ast_y   = max([accs d_batch])+3;
    end

    hold on;
    b = bar(accs,'FaceColor','flat','LineWidth',1);
    j = 1;
    for ii = 1:length(accs)
        if ~isnan(accs(ii))
            b.CData(ii,:) = facec(j,:)/255;
            j = j + 1;
        end
    end

    med = median(d_batch);
    plot([0 length(cnnLabel)],[med med],':','Color',[.5 .5 .5],'LineWidth',1.5)
    swarmchart(ones(length(d_batch),1)*(length(cnntype)+1),d_batch,7.5,'o','MarkerEdgeColor','none',...
        'MarkerFaceAlpha',0.65,'MarkerFaceColor',Chum/255,'XJitterWidth',0.4)
    boxchart(ones(length(d_batch),1)*(length(cnntype)+1),d_batch,'BoxFaceColor',Chum/255,...
        'MarkerStyle','none','BoxWidth',0.8,...
        'LineWidth',0.75,'BoxEdgeColor','k','BoxFaceAlpha',0.25)
    ylim([0 100])
    xlim([0 length(cnnLabel)+1])
    ylabel('Accuracy (%)')
    box off
    
    xticks(find(~strcmp(cnnLabel,'')))
    xticklabels(strrep(cnnLabel(~strcmp(cnnLabel,'')),'_','\_'))
    ax = gca;
    ax.TickDir = 'out';
    ax.LineWidth = 1;
    ax.FontName = 'Helvetica';
    ax.FontSize = 6;
    % stat test
    p_bf = nan(length(cnntype),1);
    wval = nan(length(cnntype),1);
    for ii = 1:length(cnntype)
        if ~isnan(accs(ii))
            [p_tmp,~,s] = signrank(d_batch,accs(ii),'method','exact');
            wval(ii) = s.signedrank;
        else
            p_tmp = NaN;
        end
        p_bf(ii) = p_tmp*sum(~isnan(accs));
        if p_bf(ii) <=0.05
            st = '*';
        else
            st = '';
        end
        text(ii,ast_y,st,'HorizontalAlignment','center','FontSize',6)
    end
    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH])
    
    saveas(gcf,[savedir '/' figlabel{type} '.png'])
    exportgraphics(gcf, [savedir '/' imtype{type} '_ac.pdf'], 'ContentType', 'vector');

    % Source data
    accs = [accs(~isnan(accs)) mean(d_batch)];
    p_bf = p_bf(~isnan(p_bf));
    wval = wval(~isnan(wval));
    t = cell(length(accs)+1,4);
    t(2:end,1) = cnnLabel(~strcmp(cnnLabel,''));
    t{1,2} = "accuracy (%)";
    t{1,3} = 'W';
    t{1,4} = 'P (Bonferroni-corrected)';
    t(2:end,2) = num2cell(accs');
    t(2:end-1,3) = num2cell(wval);
    t(2:end-1,4) = num2cell(p_bf);
    writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{type},'WriteVariableNames',false);
end

