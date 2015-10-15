import cPickle as pickle
from math import sqrt

import numpy
import scipy

from django.core.cache import get_cache as get_django_cache

from opaque_keys.edx.keys import CourseKey
import student.models
from xmodule.modulestore.django import modulestore

RECOMMENDED_COURSES_COUNT = 3

class DataNotComputed(Exception):
    pass


def for_user(user):
    """
    Recommend courses for the user based on what other similar users have liked.
    Implementation of: "Item-Based Collaborative Filtering Recommendation
    Algorithms", Sarwar et al. 2001
    http://files.grouplens.org/papers/www10_sarwar.pdf
    """
    # TODO remove me!!! Instead, run that function periodically.
    update_all_data()

    try:
        similarity_matrix = get_course_similarity_matrix()
        course_key_strings = get_course_key_string_vector()
    except DataNotComputed:
        # TODO log error
        return []

    # import ipdb; ipdb.set_trace()
    enrollments = get_user_enrollment_vector(user, course_key_strings)
    user_has_one_enrollment = any(enrollments)
    if user_has_one_enrollment:
        enrollment_scores = enrollments
    else:
        # For the sake of score computation, we suggest the most popular
        # courses to users that have not registered to any course 
        enrollment_scores = [1] * len(course_key_strings)

    affinities = []
    for course_index, course_key_string in enumerate(course_key_strings):
        if enrollments[course_index] > 0:
            # Don't compute affinity of courses already registered to
            continue
        course_similarities = similarity_matrix[course_index, :]
        norm1 = numpy.absolute(course_similarities).sum()
        affinity = course_similarities.dot(enrollment_scores)
        if norm1 > 0:
            affinity /= norm1
        affinities.append((affinity, course_key_string))

    affinities = sorted(affinities)[:RECOMMENDED_COURSES_COUNT]
    return [modulestore().get_course(CourseKey.from_string(c_id)) for a, c_id in affinities if a > 0]

def get_user_enrollment_vector(user, course_key_strings):
    enrollments = [0] * len(course_key_strings)
    for enrollment in student.models.CourseEnrollment.enrollments_for_user(user):
        course_key_string = unicode(enrollment.course_id)
        # TODO slow?
        try:
            course_index = course_key_strings.index(course_key_string)
        except ValueError:
            continue
        enrollments[course_index] = 1
    return enrollments

def update_all_data():
    update_course_key_string_vector()
    course_key_strings = get_course_key_string_vector()
    update_course_vectors(course_key_strings)
    update_course_similarity_matrix(course_key_strings)

def get_course_key_string_vector():
    pickled = get_cache().get("course-ids")
    if not pickled:
        raise DataNotComputed()
    return pickle.loads(pickled)

def update_course_key_string_vector():
    course_key_strings = compute_course_key_string_vector()
    get_cache().set("course-ids", pickle.dumps(course_key_strings))

def compute_course_key_string_vector():
    return [unicode(c.id) for c in modulestore().get_courses()]

def get_course_similarity_matrix():
    pickled = get_cache().get("course-similarity-matrix")
    if not pickled:
        raise DataNotComputed()
    return pickle.loads(pickled)

def update_course_similarity_matrix(course_key_strings):
    matrix = compute_course_similarity_matrix(course_key_strings)
    pickled = pickle.dumps(matrix)
    get_cache().set("course-similarity-matrix", pickled)

def compute_course_similarity_matrix(course_key_strings):
    matrix = scipy.eye(len(course_key_strings))
    for i in range(0, len(course_key_strings)):
        vec_i = get_course_vector(course_key_strings[i])
        if vec_i is not None:
            for j in range(i + 1, len(course_key_strings)):
                vec_j = get_course_vector(course_key_strings[j])
                if vec_j is not None:
                    scalar = numpy.dot(vec_i, vec_j.transpose())[0, 0]
                    # Note that this norm only works for matrices with (0, 1) values
                    norm = sqrt(vec_i.sum() * vec_j.sum()) if scalar != 0 else 1
                    similarity = scalar / norm
                    matrix[i, j] = similarity
                    matrix[j, i] = similarity
    return matrix

def get_course_vector(course_key_string):
    # Load course vector from cache
    # TODO recompute cache
    pickled = get_cache().get(course_key_string)
    if not pickled:
        raise DataNotComputed()
    return pickle.loads(pickled)

def update_course_vectors(course_key_strings):
    # TODO what if there is no user?
    max_user_id = student.models.User.objects.order_by('-pk')[0].pk
    for course_key_string in course_key_strings:
        vector = compute_course_vector(course_key_string, max_user_id=max_user_id)
        pickled = pickle.dumps(vector)
        get_cache().set(course_key_string, pickled)

def compute_course_vector(course_key_string, max_user_id=None):
    enrollments = student.models.CourseEnrollment.objects.filter(
        course_id=CourseKey.from_string(course_key_string)
    )
    if max_user_id is not None:
        enrollments = enrollments.filter(user__id__lt=max_user_id)
    enrollments = enrollments.order_by("user__id")

    enrollment_count = enrollments.count()
    if enrollment_count == 0:
        return None
    data = [1] * enrollment_count
    rows = [0] * enrollment_count
    cols = [e.user.id for e in enrollments]
    return scipy.sparse.csr_matrix((data, (rows, cols)),
                                   shape=(1, max_user_id + 1),
                                   dtype=numpy.int8)

def get_cache():
    return get_django_cache("recommendations")
