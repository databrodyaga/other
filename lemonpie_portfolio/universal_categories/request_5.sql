
SELECT 
    p.id, 
    p.account_id,
    p.contractor_id,
    p.date,
    p.amount AS payments_amount,
    p.extra13 AS purpose,
    p.article_id,
    p.project_id,
    p.counterpartie_id,
    p.donor_id,
    p.robot_id,
    p.donor_cat_id,
    
    a.id AS accounts__id,
    a.user_id AS accounts__user_id,

    ar.id AS articles__id,
    ar.user_id AS articles__user_id,
    ar.parent_id AS articles__parent_id,
    ar.name AS articles__name,

    pr.id AS projects__id,
    pr.user_id AS projects__user_id,
    pr.parent_id AS projects__parent_id,
    pr.name AS projects__name,

    cp.id AS counterparties__id,
    cp.user_id AS counterparties__user_id,
    cp.parent_id AS counterparties__parent_id,
    cp.name AS counterparties__name,

    r.id AS robots__id,
    r.user_id AS robots__user_id,

    aan.user_id AS article_alternative_names__user_id

    uc.uc_id AS uc__uc_id 

FROM payments AS p 
LEFT OUTER JOIN accounts AS a ON p.account_id = a.id
LEFT OUTER JOIN articles AS ar ON p.article_id = ar.id 
LEFT OUTER JOIN projects AS pr ON p.project_id = pr.id
LEFT OUTER JOIN counterparties AS cp ON p.donor_cat_id = cp.id
LEFT OUTER JOIN robots AS r ON p.robot_id = r.id
LEFT OUTER JOIN article_alternative_names AS aan ON p.article_id = aan.article_id
LEFT OUTER JOIN uc AS uc ON p.id = uc.payment_id

WHERE p.hidden = 0 AND p.deleted_at IS NULL AND p.status = 'paid' AND p.expenditure = 'incoming';
