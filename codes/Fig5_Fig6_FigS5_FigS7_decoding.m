clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir   = [rootD '/analysis/Fig5_Fig6_FigS5_FigS7_decoding'];
if ~exist(savedir,'dir');mkdir(savedir);end
imtype    = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil' 'Sparse' 'Dense'};
cnntypes = {'convnextL'  'clip_convnextL_image' 'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'...
    'vit_l_16' 'clip_vit-l-laion_image'  'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k' ...
    'resnet50'  'resnet50-blur-st' 'resnet50-sin' ...
    'cornet-s' 'cornet-s-blur-st'};
cnnLabel  = {'convnext' 'clip_convnext' 'clip_convnext_ft'...
    'vit_l' 'clip_vit-l' 'clip_vit-l_ft'...
    'resnet50' 'resnet50-sin' 'resnet50-blur'...
    'cornet' 'cornet-blur'};
nperm = 1000;

for cnn = 1:length(cnntypes)
    cnntype = cnntypes{cnn};
    if ~exist([savedir '/' cnntype '.mat'],'file')
        layers = get_layer_names(cnntype,1);
        accs = nan(length(imtype),length(layers));
        null = nan(nperm,length(imtype),length(layers));
        for lay = 1:length(layers) 
            layer = layers{lay};
            for t = 1:length(imtype) 
                actD = sprintf([rootD '/rawdata/act/%s_%s'], cnntype, imtype{t});
                act_files = dir([actD '/*/' layer '.mat']);
                for af = 1:length(act_files)
                    load([act_files(af).folder '/' act_files(af).name]);
                    if af == 1
                        actpat_all = nan(length(act_files),length(actpat), 'single');
                    end
                    actpat_all(af, :) = actpat;
                end
                r = squareform(pdist(actpat_all,'correlation'));

                bcatId = cellfun(@(s) regexp(s, '/([^/]+)_ILSVRC', 'tokens', 'once'), {act_files.folder}, 'UniformOutput', false)';
                bcatId = findgroups(cellfun(@(c) c{1}, bcatId, 'UniformOutput', false));
                pred = nan(size(r,1),1);
                for ex = 1:size(r,2)
                    r_ex = r(ex,:);
                    r_ex(ex) = NaN;
                    r_avg = splitapply(@(x) mean(x,'omitnan'),r_ex',bcatId);
                    r_avg_tmp = r_avg;
                    r_avg_tmp(bcatId(ex)) = NaN;
                    [~,pred(ex)] = min(r_avg);
                end
                accs(t,lay) = mean(pred == bcatId,"all");
                for p = 1:nperm
                    pred = nan(size(r,1),1);
                    bcatId_null = bcatId(randperm(length(bcatId)));
                    for ex = 1:size(r,2)
                        r_ex = r(ex,:);
                        r_ex(ex) = NaN;
                        r_avg = splitapply(@(x) mean(x,'omitnan'),r_ex',bcatId_null);
                        r_avg_tmp = r_avg;
                        r_avg_tmp(bcatId_null(ex)) = NaN;
                        [~,pred(ex)] = min(r_avg);
                    end
                    null(p,t,lay) = mean(pred == bcatId_null,"all");
                end
            end
        end
        save([savedir '/' cnntype '.mat'],'accs','null','layers')
    end
end


% Plot with recog accs (Exp 1)
cnntypes = {{'convnextL'  'clip_convnextL_image' 'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'}...
    {'vit_l_16' 'clip_vit-l-laion_image'  'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k' }...
    {'resnet50'  'resnet50-blur-st' 'resnet50-sin'} ...
    {'cornet-s' 'cornet-s-blur-st'}};
cnnlabels = {{'inConvNeXt'  'zsConvNeXt' 'ftConvNeXt'}...
    {'inViT' 'zsViT'  'ftViT' }...
    {'inResNet'  'blResNet' 'stResNet'} ...
    {'inCORnet' 'blCORnet'}};
titleL = {'convNext' 'ViT-L' 'resnet50' 'cornet-s'};
marks = {{'o' 'diamond' 'square'} {'o' 'diamond' 'square'} {'o' '^' 'v'} {'o' '^'}};
imtype    = {'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};

cols = {[0 48 73;247 127 0; 214 40 40]/255 [0 48 73;247 127 0; 214 40 40]/255 ...
    [0 48 73;87 204 153; 199 125 255]/255 [0 48 73;87 204 153]/255 };
layerlab = {'actF' 'actM' 'actL' 'penu' 'recog'};
figlabel = {'Fig.6' 'Fig.S7a' 'Fig.S7b' 'Fig.S7c'};
startcell = {'A' 'E' 'I' 'M' 'Q' 'U'};

jit = [.1 0 -.1];
figW = 5.5;
figH = 3.85;
for comb = 1:length(cnntypes)
    mark    = marks{comb};
    cnntype = cnntypes{comb};
    cnnlabel = cnnlabels{comb};
    col     = cols{comb};
    i = 1;
    for v = 1:length(imtype)
        figure
        hold on

        for cnn = 1:length(cnntype)
            base = get_cnn_resp(rootD,cnntype{cnn},'Ori',1);
            dat  = get_cnn_resp(rootD,cnntype{cnn},imtype{v},1);
            if ~strcmp(imtype{v},'Ori')
                dat = (dat(base==1));
            end

            load([savedir '/' cnntype{cnn} '.mat'],'accs','layers','null')
            accs = accs*100;
            null = null*100;
            if cnn == 1
                plot([0 length(layerlab)-0.5],[1/48 1/48]*100,'--','Color',[.5 .5 .5]);
                xticklabels(layerlab)
                xticks(1:length(layerlab))
                xlim([0.8 length(layerlab)+0.5])
            end
            plot(accs(v,:),'-','Color',[col(cnn,:) .8], 'LineWidth',2)
            scatter(1:size(accs,2),accs(v,:),50,mark{cnn},'MarkerEdgeColor','none',...
                'MarkerFaceColor',col(cnn,:),'MarkerFaceAlpha',0.8)
            scatter(length(layerlab),mean(dat)*100,50,mark{cnn},'MarkerEdgeColor',col(cnn,:),'LineWidth',1,...
                'MarkerFaceColor',col(cnn,:),'MarkerFaceAlpha',0.55)
            ylim([0 100])
            p = mean(accs(v,:) < squeeze(null(:,v,:)),1);
            p_bf = p*length(layerlab);
            h = p_bf<0.05;
            text(find(h)-jit(cnn),repmat(98,1,length(find(h))),'*','HorizontalAlignment','center','FontSize',10,'Color',col(cnn,:))
            ax = gca;
            ax.TickDir = 'out';
            ax.LineWidth = 1;
            ax.FontName = 'Helvetica';
            ax.FontSize = 6;
            ax.XTickLabelRotation = 0;
            % ylabel({'Decoding/'; 'Recognition accuracy (%)'})
            if cnn == 1
                p_sd = p_bf'; % p for source data
                a_sd = accs(v,:)';
                l_sd = cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab(1:end-1),'UniformOutput',false)';
            else
                p_sd = cat(1,p_sd,p_bf');
                a_sd = cat(1,a_sd,accs(v,:)');
                l_sd = cat(1,l_sd,cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab(1:end-1),'UniformOutput',false)');
            end
        end
        rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
        set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
            'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
            'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')

        exportgraphics(gcf, [savedir '/' titleL{comb} '_' imtype{v} '.pdf'], 'ContentType', 'vector');
        saveas(gcf,[savedir '/' figlabel{comb} '_' imtype{v} '.png'])
        % Source data
        if ~strcmp(imtype{v},'Ori')
            t = cell(length(a_sd)+2,3);
            t(3:end,1) = l_sd;
            t{1,1} = imtype{v};
            t{2,2} = 'decoding accuracy (%)';
            t{2,3} = 'P (Bonferroni-corrected)';
            t(3:end,2) = num2cell(a_sd);
            t(3:end,3) = num2cell(p_sd);
            writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{comb},...
                'WriteVariableNames',false,'Range',[startcell{i} num2str(1)]);
            i = i+1;
        end
    end
end

imtype   = {'Sparse' 'Dense'};
figlabel = {'Fig.5b' 'Fig.S5c' 'Fig.S5d' 'Fig.S5e'};
startcell = {'A' 'E'};
jit = [.1 0 -.1];
figW = 5.5;
figH = 3.85;
for comb = 1:length(cnntypes) % comb = 4
    mark    = marks{comb};
    cnntype = cnntypes{comb};
    col     = cols{comb};
    cnnlabel = cnnlabels{comb};
    i = 1;
    for v = 1:length(imtype)
        figure
        hold on
        for cnn = 1:length(cnntype)
            [base,imName_all] = get_cnn_resp(rootD,cnntype{cnn},'Sil',1);
            [dat,imName]      = get_cnn_resp(rootD,cnntype{cnn},imtype{v},1);
            imName_all = cellfun(@(x) x(1:end-6),imName_all,'UniformOutput',false);
            imName = cellfun(@(x) x(1:end-7),imName,'UniformOutput',false);
            exp2Idx = ismember(imName_all,imName);
            dat = (dat(base(exp2Idx)==1));
   
            load([savedir '/' cnntype{cnn} '.mat'],'accs','layers','null')
            accs = accs(end-1:end,:)*100;
            null = null(:,end-1:end,:)*100;
            if cnn == 1
                plot([0 length(layerlab)-0.5],[1/41 1/41]*100,'--','Color',[.5 .5 .5]);
                xticklabels(layerlab)
                xticks(1:length(layerlab))
                xlim([0.8 length(layerlab)+0.5])
            end
            plot(accs(v,1:4),'-','Color',[col(cnn,:) .8], 'LineWidth',2)
            scatter(1:4,accs(v,1:4),50,mark{cnn},'MarkerEdgeColor','none',...
                'MarkerFaceColor',col(cnn,:),'MarkerFaceAlpha',0.8)
            scatter(length(layerlab),mean(dat)*100,50,mark{cnn},'MarkerEdgeColor',col(cnn,:),'LineWidth',1,...
                'MarkerFaceColor',col(cnn,:),'MarkerFaceAlpha',0.55)
            ylim([0 100])
            p = mean(accs(v,1:4) < squeeze(null(:,v,1:4)),1);
            p_bf = p*4;
            h = p_bf<0.05;
            text(find(h)-jit(cnn),repmat(98,1,length(find(h))),'*','HorizontalAlignment','center','FontSize',10,'Color',col(cnn,:))
            ax = gca;
            ax.TickDir = 'out';
            ax.LineWidth = 1;
            ax.FontName = 'Helvetica';
            ax.FontSize = 6;
            ax.XTickLabelRotation = 0;
            if cnn == 1
                p_sd = p_bf'; % p for source data
                a_sd = accs(v,1:4)';
                l_sd = cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab(1:end-1),'UniformOutput',false)';
            else
                p_sd = cat(1,p_sd,p_bf');
                a_sd = cat(1,a_sd,accs(v,1:4)');
                l_sd = cat(1,l_sd,cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab(1:end-1),'UniformOutput',false)');
            end
        end
        rectangle('Position',[1 1 figW figH], ...
          'EdgeColor','none','FaceColor','none');
        set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
            'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
            'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')
        exportgraphics(gcf, [savedir '/' titleL{comb} '_' imtype{v} '.pdf'], 'ContentType', 'vector');
        saveas(gcf,[savedir '/' figlabel{comb} '_' imtype{v} '.png'])
        t = cell(length(a_sd)+2,3);
        t(3:end,1) = l_sd;
        t{1,1} = imtype{v};
        t{2,2} = 'decoding accuracy (%)';
        t{2,3} = 'P (Bonferroni-corrected)';
        t(3:end,2) = num2cell(a_sd);
        t(3:end,3) = num2cell(p_sd);
        writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{comb},...
            'WriteVariableNames',false,'Range',[startcell{i} num2str(1)]);
        i = i+1;
    end
end