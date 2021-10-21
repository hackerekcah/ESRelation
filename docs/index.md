---
layout: default
title: Exploring Inter-Node Relations in CNNs for <br> Environmental Sound Classification
description: Hongwei Song<sup>1</sup>, Shiwen Deng<sup>2</sup>, Jiqing Han<sup>1</sup> <br> <a href="http://en.hit.edu.cn/about/overview?s=info" style="color:pink;"><sup>1</sup>Harbin Institute of Technology, China</a>  <br> <a href="http://www.hrbnu.edu.cn/" style="color:pink;"><sup>2</sup>Harbin Normal University, China</a>
show_pub: true
show_appendix: true
show_download: true
---
<h1 style="text-align:center"> Visualization Examples using ESC-50 audio </h1>
<h4> Q1: How to interpret the relation heatmaps? </h4>
>>Please see Section A4 of the <a href="./appendix.html">Appendix</a>
<h4> Q2: How to interpret visualizations for ambient and impulsive style audio? </h4>
>>Please see Section A5 of the <a href="./appendix.html">Appendix</a>
<h4> Q3: What does heatmap activation mean in silence part? </h4>
>>Please see Section A6 of the <a href="./appendix.html">Appendix</a>
<table width="200%">
    <tr>
        <th>Audio</th>
        <th>Relation Heatmap</th>
        <th>Relation Structure Heatmap</th>
    </tr>

    {% assign audio_files = site.static_files | where: "wav", true %}
    {% for wav_file in audio_files %}

        <tr>
            <td style="text-align:center">
                <h3><strong>{{ wav_file.basename }}</strong></h3>
                <audio controls>
                  <source src="{{ wav_file.path | relative_url}}" type="audio/wav">
                  Your browsers did not support audio tag.
                </audio>
            </td>
            <td>
                {% assign gif_file = wav_file.path | replace: "wavs", "gifs" |replace: wav_file.extname, ".gif" %}
                <img src="{{ gif_file | relative_url}}"/>
            </td>
            <td>
                {% assign png_file = wav_file.path | replace: "wavs", "pngs" |replace: wav_file.extname, "_structure.png" %}
                <img src="{{ png_file | relative_url}}"/>
            </td>
        </tr>
    {% endfor %}
</table>
