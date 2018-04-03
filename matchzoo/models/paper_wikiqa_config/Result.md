## Result

The results are as follows.

<table>
  <tr>
    <th width=10%, bgcolor=#999999 >Models</th> 
    <th width=20%, bgcolor=#999999>NDCG@3</th>
    <th width="20%", bgcolor=#999999>NDCG@5</th>
    <th width="20%", bgcolor=#999999>PRECISION@1</th>
    <th width="20%", bgcolor=#999999>MAP</th>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> BIGRU </td>
    <td align="center", bgcolor=#eeeeee> 0.635690 </td>
    <td align="center", bgcolor=#eeeeee> 0.690004 </td>
    <td align="center", bgcolor=#eeeeee> 0.510549 </td>
    <td align="center", bgcolor=#eeeeee> 0.647465 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> BILSTM </td>
    <td align="center", bgcolor=#eeeeee> 0.604327 </td>
    <td align="center", bgcolor=#eeeeee> 0.666213 </td>
    <td align="center", bgcolor=#eeeeee> 0.485232 </td>
    <td align="center", bgcolor=#eeeeee> 0.624780 </td>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> ESelfAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.596405 </td>
    <td align="center", bgcolor=#eeeeee> 0.655927 </td>
    <td align="center", bgcolor=#eeeeee> 0.468354 </td>
    <td align="center", bgcolor=#eeeeee> 0.611272 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> ESelfLSTMAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.606049 </td>
    <td align="center", bgcolor=#eeeeee> 0.658696 </td>
    <td align="center", bgcolor=#eeeeee> 0.476793  </td>
    <td align="center", bgcolor=#eeeeee> 0.616515 </td>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> ECrossAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.590751 </td>
    <td align="center", bgcolor=#eeeeee> 0.648090 </td>
    <td align="center", bgcolor=#eeeeee> 0.455696 </td>
    <td align="center", bgcolor=#eeeeee> 0.610558 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> ECrossLSTMAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.603962 </td>
    <td align="center", bgcolor=#eeeeee> 0.649862 </td>
    <td align="center", bgcolor=#eeeeee> 0.493671  </td>
    <td align="center", bgcolor=#eeeeee> 0.618218 </td>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> EAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.617125 </td>
    <td align="center", bgcolor=#eeeeee> 0.667854 </td>
    <td align="center", bgcolor=#eeeeee> 0.502110 </td>
    <td align="center", bgcolor=#eeeeee> 0.634342 </td>
  </tr>  -->
  <tr>
    <td align="center", bgcolor=#eeeeee> ELSTMAttention </td>
    <td align="center", bgcolor=#eeeeee> 0.644557 </td>
    <td align="center", bgcolor=#eeeeee> 0.688240 </td>
    <td align="center", bgcolor=#eeeeee> 0.510549 </td>
    <td align="center", bgcolor=#eeeeee> 0.648403 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> ANMM </td>
    <td align="center", bgcolor=#eeeeee> 0.628366 </td>
    <td align="center", bgcolor=#eeeeee> 0.675884 </td>
    <td align="center", bgcolor=#eeeeee> 0.481013 </td>
    <td align="center", bgcolor=#eeeeee> 0.630149 </td>
  </tr> 
  <tr>
    <td align="center", bgcolor=#eeeeee> ARCI </td>
    <td align="center", bgcolor=#eeeeee> 0.583519 </td>
    <td align="center", bgcolor=#eeeeee> 0.633970 </td>
    <td align="center", bgcolor=#eeeeee> 0.451477 </td>
    <td align="center", bgcolor=#eeeeee> 0.595163 </td>
  </tr> 
  <tr>
    <td align="center", bgcolor=#eeeeee> ARCII </td>
    <td align="center", bgcolor=#eeeeee> 0.536577 </td>
    <td align="center", bgcolor=#eeeeee> 0.601120 </td>
    <td align="center", bgcolor=#eeeeee> 0.447257 </td>
    <td align="center", bgcolor=#eeeeee> 0.570888 </td>
  </tr> 
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> BIMPM </td>
    <td align="center", bgcolor=#eeeeee> 0.660716 </td>
    <td align="center", bgcolor=#eeeeee> 0.715140 </td>
    <td align="center", bgcolor=#eeeeee> 0.540084 </td>
    <td align="center", bgcolor=#eeeeee> 0.668650 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> CDSSM </td>
    <td align="center", bgcolor=#eeeeee> 0.419517 </td>
    <td align="center", bgcolor=#eeeeee> 0.516156 </td>
    <td align="center", bgcolor=#eeeeee> 0.253165 </td>
    <td align="center", bgcolor=#eeeeee> 0.462335 </td>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> DRMM_TKS </td>
    <td align="center", bgcolor=#eeeeee> 0.655163 </td>
    <td align="center", bgcolor=#eeeeee> 0.703279 </td>
    <td align="center", bgcolor=#eeeeee> 0.523207 </td>
    <td align="center", bgcolor=#eeeeee> 0.663941 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> DRMM </td>
    <td align="center", bgcolor=#eeeeee> 0.626560 </td>
    <td align="center", bgcolor=#eeeeee> 0.674424 </td>
    <td align="center", bgcolor=#eeeeee> 0.497890 </td>
    <td align="center", bgcolor=#eeeeee> 0.635241 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> DSSM </td>
    <td align="center", bgcolor=#eeeeee> 0.572077 </td>
    <td align="center", bgcolor=#eeeeee> 0.633932 </td>
    <td align="center", bgcolor=#eeeeee> 0.421941 </td>
    <td align="center", bgcolor=#eeeeee> 0.590641 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> DUET </td>
    <td align="center", bgcolor=#eeeeee> 0.616912 </td>
    <td align="center", bgcolor=#eeeeee> 0.668997 </td>
    <td align="center", bgcolor=#eeeeee> 0.502110 </td>
    <td align="center", bgcolor=#eeeeee> 0.632204 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> MatchPyramid </td>
    <td align="center", bgcolor=#eeeeee> 0.610435 </td>
    <td align="center", bgcolor=#eeeeee> 0.669629 </td>
    <td align="center", bgcolor=#eeeeee> 0.459916  </td>
    <td align="center", bgcolor=#eeeeee> 0.620812 </td>
  </tr>
<!--   <tr>
    <td align="center", bgcolor=#eeeeee> MVLSTM </td>
    <td align="center", bgcolor=#eeeeee> 0.624306 </td>
    <td align="center", bgcolor=#eeeeee> 0.672264 </td>
    <td align="center", bgcolor=#eeeeee> 0.510549  </td>
    <td align="center", bgcolor=#eeeeee> 0.640767 </td>
  </tr> -->
  <tr>
    <td align="center", bgcolor=#eeeeee> KNRM </td>
    <td align="center", bgcolor=#eeeeee> 0.553560 </td>
    <td align="center", bgcolor=#eeeeee> 0.616591 </td>
    <td align="center", bgcolor=#eeeeee> 0.396624  </td>
    <td align="center", bgcolor=#eeeeee> 0.565658 </td>
  </tr>
</table>
